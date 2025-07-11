import FeatureDashboard from '@/app/[modelId]/[layer]/[index]/feature-dashboard';
import { ANT_MODEL_ID_TO_NEURONPEDIA_MODEL_ID, CLTGraph, CLTGraphNode } from '@/app/[modelId]/graph/utils';
import { useGlobalContext } from '@/components/provider/global-provider';
import { Button } from '@/components/shadcn/button';
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/shadcn/hover-card';
import { LoadingSquare } from '@/components/svg/loading-square';
import { BOS_TOKENS } from '@/lib/utils/activations';
import { SteerLogitFeature } from '@/lib/utils/graph';
import {
  STEER_STRENGTH_ADDED_MULTIPLIER_GRAPH,
  STEER_STRENGTH_ADDED_MULTIPLIER_MAX,
  STEER_STRENGTH_ADDED_MULTIPLIER_MIN,
} from '@/lib/utils/steer';
import { NeuronWithPartialRelations } from '@/prisma/generated/zod';
import { InfoCircledIcon } from '@radix-ui/react-icons';
import * as Slider from '@radix-ui/react-slider';
import { MousePointerClick, Trash2 } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

export type SteeredPositionIdentifier = {
  modelId: keyof typeof ANT_MODEL_ID_TO_NEURONPEDIA_MODEL_ID;
  layer: number;
  index: number;
  tokenActivePosition: number;
};

// Each NodeToSteer is a single feature for a single active "original" token position.
// Each node can have multiple positions that it's steered at.
export default function NodeToSteer({
  node,
  sourceId,
  nodeSteerIdentifier,
  label,
  selectedGraph,
  steeredPositions,
  setSteeredPositions,
  isSteered,
}: {
  node: CLTGraphNode;
  sourceId: string;
  nodeSteerIdentifier: SteeredPositionIdentifier;
  label: string;
  selectedGraph: CLTGraph;
  steeredPositions: SteerLogitFeature[];
  setSteeredPositions: (features: SteerLogitFeature[]) => void;
  isSteered: (identifier: SteeredPositionIdentifier) => boolean;
}) {
  const { setFeatureModalFeature, setFeatureModalOpen } = useGlobalContext();
  const scrollIntoViewRef = useRef<HTMLDivElement>(null);
  const [hoveredFeature, setHoveredFeature] = useState<NeuronWithPartialRelations | undefined>();

  function findSteeredPositionByPosition(position: number) {
    return steeredPositions.find(
      (f) =>
        f.layer === nodeSteerIdentifier.layer &&
        f.index === nodeSteerIdentifier.index &&
        f.steer_position === position &&
        f.token_active_position === nodeSteerIdentifier.tokenActivePosition,
    );
  }

  function findSteeredPositionSteerGeneratedTokens() {
    return steeredPositions.find(
      (f) =>
        f.layer === nodeSteerIdentifier.layer &&
        f.index === nodeSteerIdentifier.index &&
        f.steer_generated_tokens &&
        f.token_active_position === nodeSteerIdentifier.tokenActivePosition,
    );
  }

  function removeSteeredPosition() {
    setSteeredPositions(
      steeredPositions.filter(
        (f) =>
          !(
            f.layer === nodeSteerIdentifier.layer &&
            f.index === nodeSteerIdentifier.index &&
            f.token_active_position === nodeSteerIdentifier.tokenActivePosition
          ),
      ),
    );
  }

  function setSteeredPositionDeltaSteerGeneratedTokens(delta: number) {
    const ablate = delta === 0;
    const newDelta = ablate ? null : delta;
    if (findSteeredPositionSteerGeneratedTokens()) {
      setSteeredPositions(
        steeredPositions.map((f) =>
          f.layer === nodeSteerIdentifier.layer &&
          f.index === nodeSteerIdentifier.index &&
          f.steer_generated_tokens &&
          f.token_active_position === nodeSteerIdentifier.tokenActivePosition
            ? { ...f, delta: newDelta, ablate }
            : f,
        ),
      );
    } else {
      setSteeredPositions([
        ...steeredPositions,
        {
          layer: nodeSteerIdentifier.layer,
          index: nodeSteerIdentifier.index,
          delta: newDelta,
          ablate,
          steer_generated_tokens: true,
          token_active_position: nodeSteerIdentifier.tokenActivePosition,
        },
      ]);
    }
  }

  function setSteeredPositionDeltaByPosition(position: number, delta: number) {
    const ablate = delta === 0;
    const newDelta = ablate ? null : delta;
    const feature = findSteeredPositionByPosition(position);
    // feature not currently steered, add the steer at the specified position and delta, and also make it steer generated tokens
    if (!feature) {
      setSteeredPositions([
        ...steeredPositions,
        {
          layer: nodeSteerIdentifier.layer,
          index: nodeSteerIdentifier.index,
          delta: newDelta,
          steer_position: position,
          ablate,
          steer_generated_tokens: false,
          token_active_position: nodeSteerIdentifier.tokenActivePosition,
        },
      ]);
      return;
    }
    // feature is currently steered at this position, update the delta
    setSteeredPositions(
      steeredPositions.map((f) =>
        f.layer === nodeSteerIdentifier.layer &&
        f.index === nodeSteerIdentifier.index &&
        f.steer_position === position &&
        f.token_active_position === nodeSteerIdentifier.tokenActivePosition
          ? { ...f, delta: newDelta, ablate }
          : f,
      ),
    );
  }

  // number is the delta they all have in common, false means not all the same, null = ablate
  function allTokensHaveSameDelta(): number | false | null {
    if (steeredPositions.length <= 1) {
      return false;
    }
    // find the first delta for this feature at any position
    const firstDelta = steeredPositions.find(
      (f) => f.layer === nodeSteerIdentifier.layer && f.index === nodeSteerIdentifier.index,
    )?.delta;
    if (firstDelta === undefined) {
      return false;
    }
    for (let i = 0; i < selectedGraph.metadata.prompt_tokens.length; i += 1) {
      // don't check BOS
      if (BOS_TOKENS.includes(selectedGraph.metadata.prompt_tokens[i])) {
        // eslint-disable-next-line no-continue
        continue;
      }
      const feature = findSteeredPositionByPosition(i);
      if (feature?.delta !== firstDelta) {
        return false;
      }
    }
    // check the "generated" too
    const generatedFeature = findSteeredPositionSteerGeneratedTokens();
    if (generatedFeature?.delta !== firstDelta) {
      return false;
    }
    return firstDelta;
  }

  function setAllTokensDelta(delta: number) {
    let newSteeredPositionFeatures = steeredPositions;
    // remove all steered positions for this node
    newSteeredPositionFeatures = newSteeredPositionFeatures.filter(
      (f) =>
        !(
          f.layer === nodeSteerIdentifier.layer &&
          f.index === nodeSteerIdentifier.index &&
          f.token_active_position === nodeSteerIdentifier.tokenActivePosition
        ),
    );
    // then set this delta for every position
    for (let i = 0; i < selectedGraph.metadata.prompt_tokens.length; i += 1) {
      // don't steer BOS
      if (BOS_TOKENS.includes(selectedGraph.metadata.prompt_tokens[i])) {
        // eslint-disable-next-line no-continue
        continue;
      }
      newSteeredPositionFeatures.push({
        layer: nodeSteerIdentifier.layer,
        index: nodeSteerIdentifier.index,
        token_active_position: nodeSteerIdentifier.tokenActivePosition,
        delta: delta === 0 ? null : delta,
        ablate: delta === 0,
        steer_position: i,
        steer_generated_tokens: false,
      });
    }
    // then add the "generated" too
    newSteeredPositionFeatures.push({
      layer: nodeSteerIdentifier.layer,
      index: nodeSteerIdentifier.index,
      delta: delta === 0 ? null : delta,
      ablate: delta === 0,
      steer_generated_tokens: true,
      token_active_position: nodeSteerIdentifier.tokenActivePosition,
    });
    setSteeredPositions(newSteeredPositionFeatures);
  }

  function getTopActivationValue() {
    if (node.featureDetailNP?.activations?.length && node.featureDetailNP.activations.length > 0) {
      return node.featureDetailNP.activations[0].maxValue || 0;
    }
    return 0;
  }

  // finds the top activation for this feature and multiplies it by the multipler.
  function getDeltaToAddForMultiplier(multiplier: number) {
    // get the top activation
    return getTopActivationValue() * multiplier;
  }

  // Scroll to the position when the component mounts or when steered features change
  useEffect(() => {
    // for some reason the scroll doesn't set to the correct position if a timeout is not set
    setTimeout(() => {
      if (scrollIntoViewRef.current && isSteered(nodeSteerIdentifier)) {
        scrollIntoViewRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
      }
    }, 1);
  }, [isSteered, nodeSteerIdentifier]);

  return (
    <div key={node.nodeId} className="flex w-full flex-col gap-y-1 rounded-md py-0.5 pl-1">
      <div className="flex flex-row items-center gap-x-3 text-[10px]">
        <Button
          onClick={() => {
            if (isSteered(nodeSteerIdentifier)) {
              removeSteeredPosition();
            } else {
              setSteeredPositionDeltaByPosition(
                node.ctx_idx,
                getDeltaToAddForMultiplier(STEER_STRENGTH_ADDED_MULTIPLIER_GRAPH),
              );
            }
          }}
          className={`h-6 w-20 rounded-full border text-[9px] font-medium uppercase ${
            isSteered(nodeSteerIdentifier)
              ? 'border-red-600 bg-red-50 text-red-600 hover:bg-red-100'
              : 'border-sky-700 bg-white text-sky-800 hover:bg-sky-200'
          }`}
        >
          {isSteered(nodeSteerIdentifier) ? 'Remove' : 'Steer'}
        </Button>
        <div className="flex flex-1 basis-1/2 flex-col gap-y-[1px]">
          <div className="-mt-[1px] line-clamp-1 flex-1 text-xs" title={label}>
            {label}
          </div>
          <div className="flex w-full flex-row items-center justify-start text-[9px] font-medium leading-none text-slate-400 group-hover:text-slate-500">
            Activation <span className="px-1 font-mono text-sky-700">{node.activation?.toFixed(2)}</span> at{' '}
            <div className="mx-1 rounded-sm bg-slate-200 px-0.5 py-0.5 font-mono text-slate-600">
              {selectedGraph.metadata.prompt_tokens[nodeSteerIdentifier.tokenActivePosition]
                .toString()
                .replaceAll(' ', '\u00A0')
                .replaceAll('\n', '↵')}
            </div>{' '}
            position {nodeSteerIdentifier.tokenActivePosition}
          </div>
        </div>

        <div className="flex flex-col gap-x-1 pl-0">
          <HoverCard openDelay={0} closeDelay={0}>
            <HoverCardTrigger asChild>
              <Button
                onMouseEnter={() => {
                  // if hoveredFeature has same source and index, don't load it again
                  if (
                    hoveredFeature &&
                    hoveredFeature.layer === sourceId &&
                    hoveredFeature.index === nodeSteerIdentifier.index.toString()
                  ) {
                    return;
                  }

                  // if it's not the same, reset it
                  setHoveredFeature(undefined);

                  fetch(`/api/feature/${nodeSteerIdentifier.modelId}/${sourceId}/${nodeSteerIdentifier.index}`, {
                    method: 'GET',
                    headers: { 'Content-Type': 'application/json' },
                  })
                    .then((response) => response.json())
                    .then((n: NeuronWithPartialRelations) => {
                      setHoveredFeature(n);
                    })
                    .catch((error) => {
                      console.error(`error submitting getting rest of neuron: ${error}`);
                    });
                }}
                onClick={() => {
                  if (
                    hoveredFeature &&
                    hoveredFeature.layer === sourceId &&
                    hoveredFeature.index === nodeSteerIdentifier.index.toString() &&
                    hoveredFeature.activations
                  ) {
                    // if we already loaded this from hover, then just set it and open it
                    setFeatureModalFeature(hoveredFeature);
                  } else {
                    setFeatureModalFeature({
                      modelId: nodeSteerIdentifier.modelId,
                      layer: sourceId,
                      index: nodeSteerIdentifier.index.toString(),
                    } as NeuronWithPartialRelations);
                  }
                  setFeatureModalOpen(true);
                }}
                className="group relative mr-0 flex h-7 min-w-[95px] shrink-0 flex-row items-center justify-start whitespace-nowrap rounded-md border border-transparent bg-slate-200 px-0 py-[6px] text-[8.5px] font-medium leading-none text-slate-500 shadow-none hover:border-sky-700 hover:bg-sky-200 hover:text-sky-700"
              >
                <div className="flex flex-col items-start justify-start gap-y-[2px] pl-[11px] text-left font-mono font-medium">
                  <div className="">LAYER {nodeSteerIdentifier.layer}</div>
                  <div className="">INDEX {nodeSteerIdentifier.index}</div>
                </div>
                <InfoCircledIcon className="absolute right-2 h-3 w-3 text-slate-500 group-hover:text-sky-700" />
              </Button>
            </HoverCardTrigger>
            <HoverCardContent className="h-[386px] max-h-[386px] min-h-[386px] w-[512px] min-w-[512px] max-w-[512px] overflow-y-hidden border-0 bg-white p-0">
              {hoveredFeature?.activations && hoveredFeature?.activations?.length > 0 ? (
                <div className="-mt-2 h-full w-full">
                  <FeatureDashboard
                    key={`${hoveredFeature?.modelId}-${hoveredFeature?.layer}-${hoveredFeature?.index}`}
                    initialNeuron={hoveredFeature}
                    embed
                    forceMiniStats
                    activationMarkerValue={node.activation}
                  />
                </div>
              ) : (
                <div className="flex h-full w-full items-center justify-center rounded-lg border">
                  <LoadingSquare className="h-6 w-6" />
                </div>
              )}
            </HoverCardContent>
          </HoverCard>
        </div>
      </div>
      {isSteered(nodeSteerIdentifier) && (
        <div className="mt-0.5 flex w-full flex-row items-center gap-x-1">
          <div className="mb-1 flex w-full flex-col">
            <div className="flex max-w-full flex-1 flex-row items-start gap-x-1.5 pb-1">
              <div className="flex min-w-11 flex-col items-center justify-center gap-y-1 rounded bg-slate-200/50 py-2">
                <Slider.Root
                  orientation="vertical"
                  defaultValue={[0]}
                  min={STEER_STRENGTH_ADDED_MULTIPLIER_MIN}
                  max={STEER_STRENGTH_ADDED_MULTIPLIER_MAX}
                  step={0.1}
                  value={(() => {
                    const delta = allTokensHaveSameDelta();
                    if (delta === null || delta === false) {
                      return [0];
                    }
                    return [Number((delta / getTopActivationValue()).toFixed(1)) || 0];
                  })()}
                  onValueChange={(value) => {
                    setAllTokensDelta(getDeltaToAddForMultiplier(value[0]));
                  }}
                  className={`group relative flex h-24 w-2 items-center justify-center overflow-visible ${
                    allTokensHaveSameDelta() === false ? 'opacity-50 hover:opacity-70' : 'cursor-pointer'
                  }`}
                >
                  <Slider.Track className="relative h-full w-[4px] grow cursor-pointer rounded-full border border-sky-600 bg-sky-600 disabled:border-slate-300 disabled:bg-slate-100 group-hover:bg-sky-700">
                    <Slider.Range className="absolute h-full rounded-full" />
                  </Slider.Track>
                  <Slider.Thumb
                    onClick={() => {
                      if (!allTokensHaveSameDelta()) {
                        setAllTokensDelta(0);
                      }
                    }}
                    className="relative flex h-5 w-9 cursor-pointer select-none items-center justify-center overflow-visible rounded-full border border-sky-700 bg-white text-[10px] font-medium leading-none text-sky-700 shadow disabled:border-slate-300 disabled:bg-slate-100 group-hover:bg-sky-100"
                  >
                    {allTokensHaveSameDelta() === false ? (
                      <MousePointerClick className="h-3.5 w-3.5" />
                    ) : allTokensHaveSameDelta() === null ? (
                      <span className="text-[7px] font-bold">ABLATE</span>
                    ) : (
                      // @ts-ignore
                      `${allTokensHaveSameDelta() ? (allTokensHaveSameDelta() > 0 ? '+' : '') : ''}${(allTokensHaveSameDelta() / getTopActivationValue()).toFixed(1) || '0'}×`
                    )}
                  </Slider.Thumb>
                </Slider.Root>
                <div className="mt-0.5 flex h-5 flex-col items-center justify-center gap-y-[1px] rounded px-1 py-0 text-center text-[7px] font-medium leading-none text-slate-500">
                  <div>ALL</div>
                  <div>TOKENS</div>
                </div>
              </div>
              <div className="forceShowScrollBarHorizontal flex max-w-full flex-row items-end overflow-x-scroll px-2 py-2 pb-0.5">
                {selectedGraph?.metadata.prompt_tokens.map((token, i) => (
                  <div
                    key={`${node.nodeId}-${i}`}
                    ref={node.ctx_idx === i ? scrollIntoViewRef : undefined}
                    className={`mx-1.5 min-w-fit flex-col items-center justify-center gap-y-1 ${
                      BOS_TOKENS.includes(token) ? 'hidden' : 'flex'
                    }`}
                  >
                    <Slider.Root
                      orientation="vertical"
                      defaultValue={[0]}
                      min={STEER_STRENGTH_ADDED_MULTIPLIER_MIN}
                      max={STEER_STRENGTH_ADDED_MULTIPLIER_MAX}
                      step={0.1}
                      value={[(findSteeredPositionByPosition(i)?.delta || 0) / getTopActivationValue() || 0]}
                      onValueChange={(value) => {
                        setSteeredPositionDeltaByPosition(i, getDeltaToAddForMultiplier(value[0]));
                      }}
                      className={`group relative flex h-24 w-2 items-center justify-center overflow-visible ${
                        !findSteeredPositionByPosition(i) ? 'opacity-50 hover:opacity-70' : 'cursor-pointer'
                      }`}
                    >
                      <Slider.Track className="relative h-full w-[4px] grow cursor-pointer rounded-full border border-sky-600 bg-sky-600 disabled:border-slate-300 disabled:bg-slate-100 group-hover:bg-sky-700">
                        <Slider.Range className="absolute h-full rounded-full" />
                      </Slider.Track>
                      <Slider.Thumb
                        onClick={() => {
                          if (!findSteeredPositionByPosition(i)) {
                            setSteeredPositionDeltaByPosition(i, 0);
                          }
                        }}
                        className="relative flex h-5 w-9 cursor-pointer select-none items-center justify-center overflow-visible rounded-full border border-sky-700 bg-white text-[10px] font-medium leading-none text-sky-700 shadow disabled:border-slate-300 disabled:bg-slate-100 group-hover:bg-sky-100"
                      >
                        {!findSteeredPositionByPosition(i) ? (
                          <MousePointerClick className="h-3.5 w-3.5" />
                        ) : findSteeredPositionByPosition(i)?.ablate ? (
                          <span className="text-[7px] font-bold">ABLATE</span>
                        ) : (
                          // @ts-ignore
                          `${findSteeredPositionByPosition(i)?.delta ? (findSteeredPositionByPosition(i)?.delta > 0 ? '+' : '') : ''}${((findSteeredPositionByPosition(i)?.delta || 0) / getTopActivationValue()).toFixed(1) || '0'}×`
                        )}
                      </Slider.Thumb>
                    </Slider.Root>
                    <div className="mt-0.5 h-5 rounded bg-slate-200 px-1 py-0.5 font-mono text-[8.5px] text-slate-700">
                      {token.toString().replaceAll(' ', '\u00A0').replaceAll('\n', '↵')}
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      className={`flex h-6 w-6 min-w-6 rounded p-0 text-red-700 hover:bg-red-100 ${
                        findSteeredPositionByPosition(i) ? 'text-red-700 hover:bg-red-100' : 'text-slate-400'
                      }`}
                      disabled={!findSteeredPositionByPosition(i)}
                      onClick={() =>
                        setSteeredPositions(
                          steeredPositions.filter(
                            (f) =>
                              !(
                                f.layer === nodeSteerIdentifier.layer &&
                                f.index === nodeSteerIdentifier.index &&
                                f.steer_position === i &&
                                f.token_active_position === nodeSteerIdentifier.tokenActivePosition
                              ),
                          ),
                        )
                      }
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                ))}
              </div>
              <div className="flex w-11 min-w-11 max-w-11 flex-col items-center justify-center gap-y-1 rounded bg-slate-200/50 py-2">
                <Slider.Root
                  orientation="vertical"
                  defaultValue={[0]}
                  min={STEER_STRENGTH_ADDED_MULTIPLIER_MIN}
                  max={STEER_STRENGTH_ADDED_MULTIPLIER_MAX}
                  step={0.1}
                  value={[(findSteeredPositionSteerGeneratedTokens()?.delta || 0) / getTopActivationValue() || 0]}
                  onValueChange={(value) => {
                    setSteeredPositionDeltaSteerGeneratedTokens(getDeltaToAddForMultiplier(value[0]));
                  }}
                  className={`group relative flex h-24 w-2 items-center justify-center overflow-visible ${
                    !findSteeredPositionSteerGeneratedTokens() ? 'opacity-50 hover:opacity-70' : 'cursor-pointer'
                  }`}
                >
                  <Slider.Track className="relative h-full w-[4px] grow cursor-pointer rounded-full border border-sky-600 bg-sky-600 disabled:border-slate-300 disabled:bg-slate-100 group-hover:bg-sky-700">
                    <Slider.Range className="absolute h-full rounded-full" />
                  </Slider.Track>
                  <Slider.Thumb
                    onClick={() => {
                      if (!findSteeredPositionSteerGeneratedTokens()) {
                        setSteeredPositionDeltaSteerGeneratedTokens(0);
                      }
                    }}
                    className="relative flex h-5 w-9 cursor-pointer select-none items-center justify-center overflow-visible rounded-full border border-sky-700 bg-white text-[10px] font-medium leading-none text-sky-700 shadow disabled:border-slate-300 disabled:bg-slate-100 group-hover:bg-sky-100"
                  >
                    {!findSteeredPositionSteerGeneratedTokens() ? (
                      <MousePointerClick className="h-3.5 w-3.5" />
                    ) : findSteeredPositionSteerGeneratedTokens()?.ablate ? (
                      <span className="text-[7px] font-bold">ABLATE</span>
                    ) : (
                      // @ts-ignore
                      `${findSteeredPositionSteerGeneratedTokens()?.delta ? (findSteeredPositionSteerGeneratedTokens()?.delta > 0 ? '+' : '') : ''}${((findSteeredPositionSteerGeneratedTokens()?.delta || 0) / getTopActivationValue()).toFixed(1) || '0'}×`
                    )}
                  </Slider.Thumb>
                </Slider.Root>
                <div className="mt-0.5 flex h-5 flex-col items-center justify-center gap-y-[1px] rounded px-1 py-0 text-center text-[7px] font-medium leading-none text-slate-500">
                  <div>NEW</div>
                  <div>TOKENS</div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className={`flex h-6 w-6 min-w-6 rounded p-0 text-red-700 hover:bg-red-100 ${
                    findSteeredPositionSteerGeneratedTokens() ? 'text-red-700 hover:bg-red-100' : 'text-slate-400'
                  }`}
                  disabled={!findSteeredPositionSteerGeneratedTokens()}
                  onClick={() =>
                    setSteeredPositions(
                      steeredPositions.filter(
                        (f) =>
                          !(
                            f.layer === nodeSteerIdentifier.layer &&
                            f.index === nodeSteerIdentifier.index &&
                            f.steer_generated_tokens
                          ),
                      ),
                    )
                  }
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
