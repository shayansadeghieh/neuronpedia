import FeatureDashboard from '@/app/[modelId]/[layer]/[index]/feature-dashboard';
import CustomTooltip from '@/components/custom-tooltip';
import { useGlobalContext } from '@/components/provider/global-provider';
import { useGraphModalContext } from '@/components/provider/graph-modal-provider';
import { useGraphContext } from '@/components/provider/graph-provider';
import { Button } from '@/components/shadcn/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/shadcn/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/shadcn/dialog';
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/shadcn/hover-card';
import { LoadingSquare } from '@/components/svg/loading-square';
import { BOS_TOKENS } from '@/lib/utils/activations';
import { SteerLogitFeature, SteerLogitsRequest, SteerResponse, SteerResponseLogitsByToken } from '@/lib/utils/graph';
import {
  STEER_FREEZE_ATTENTION,
  STEER_FREQUENCY_PENALTY_GRAPH,
  STEER_FREQUENCY_PENALTY_MAX,
  STEER_FREQUENCY_PENALTY_MIN,
  STEER_N_COMPLETION_TOKENS_GRAPH,
  STEER_N_COMPLETION_TOKENS_GRAPH_MAX,
  STEER_SEED,
  STEER_STRENGTH_MAX,
  STEER_STRENGTH_MIN,
  STEER_TEMPERATURE_GRAPH,
  STEER_TEMPERATURE_MAX,
} from '@/lib/utils/steer';
import { NeuronWithPartialRelations } from '@/prisma/generated/zod';
import { ResetIcon } from '@radix-ui/react-icons';
import * as Slider from '@radix-ui/react-slider';
import { ArrowUpRightFromSquare, Joystick, MousePointerClick, Trash2 } from 'lucide-react';
import { useEffect, useState } from 'react';
import {
  ANT_MODEL_ID_TO_NEURONPEDIA_MODEL_ID,
  CLTGraph,
  CLTGraphNode,
  getIndexFromAnthropicFeatureId,
  getLayerFromAnthropicFeatureId,
  MODEL_TO_SOURCESET_ID,
  nodeTypeHasFeatureDetail,
} from '../utils';

function TokenTooltip({ logitsByToken }: { logitsByToken: SteerResponseLogitsByToken }) {
  return (
    <div className="flex flex-wrap items-end gap-x-0 gap-y-[0px]">
      {logitsByToken.map((token, index) => {
        if (token.top_logits.length === 0) {
          return (
            <span
              key={`${token.token}-${index}`}
              className="cursor-default py-[3px] font-mono text-[11px] text-slate-800"
            >
              {token.token.toString().replaceAll(' ', '\u00A0').replaceAll('\n', '↵')}
            </span>
          );
        }
        return (
          <CustomTooltip
            key={`${token.token}-${index}`}
            trigger={
              <span
                className={`cursor-pointer rounded px-[3px] py-[3px] font-mono text-[11px] text-slate-800 transition-all ${
                  token.top_logits.length === 0 ? 'bg-slate-100' : 'ml-[3px] bg-sky-100 hover:bg-sky-200'
                }`}
              >
                {token.token.toString().replaceAll(' ', '\u00A0').replaceAll('\n', '↵')}
              </span>
            }
          >
            {token.top_logits.length === 0 ? (
              <div>{token.token.toString().replaceAll(' ', '\u00A0').replaceAll('\n', '↵')}</div>
            ) : (
              <div className="flex w-full min-w-[160px] flex-col gap-y-0.5">
                <div className="mb-2 flex flex-row justify-between gap-x-3 border-b border-slate-300 pb-1">
                  <span className="text-xs text-slate-500">Token</span>
                  <span className="text-xs text-slate-500">Probability</span>
                </div>
                {token.top_logits.map((logit) => (
                  <div key={logit.token} className="flex flex-row items-center justify-between gap-x-1 font-mono">
                    <span className="rounded bg-slate-200 px-[3px] py-[2px] text-[11px] text-slate-700">
                      {logit.token.toString().replaceAll(' ', '\u00A0').replaceAll('\n', '↵')}
                    </span>
                    <span className="text-[11px] text-slate-600">{logit.prob.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            )}
          </CustomTooltip>
        );
      })}
    </div>
  );
}

function NodeToSteer({
  node,
  label,
  selectedGraph,
  steerLogitFeatures,
  setSteerLogitFeatures,
}: {
  node: CLTGraphNode;
  label: string;
  selectedGraph: CLTGraph;
  steerLogitFeatures: SteerLogitFeature[];
  setSteerLogitFeatures: (features: SteerLogitFeature[]) => void;
}) {
  const { setFeatureModalFeature, setFeatureModalOpen } = useGlobalContext();

  function findSelectedFeature(layer: number, index: number) {
    // find ANY feature with this layer and index
    return steerLogitFeatures.find((f) => f.layer === layer && f.index === index);
  }

  const modelId = ANT_MODEL_ID_TO_NEURONPEDIA_MODEL_ID[
    selectedGraph?.metadata.scan as keyof typeof ANT_MODEL_ID_TO_NEURONPEDIA_MODEL_ID
  ] as keyof typeof ANT_MODEL_ID_TO_NEURONPEDIA_MODEL_ID;

  const layer = getLayerFromAnthropicFeatureId(modelId, node.feature);
  const sourceId = `${layer}-${MODEL_TO_SOURCESET_ID[modelId]}`;
  const index = getIndexFromAnthropicFeatureId(modelId, node.feature);

  const lastPosition = selectedGraph.metadata.prompt_tokens.length - 1;

  function setFeatureDelta(featLayer: number, featIndex: number, delta: number) {
    const ablate = delta === 0;
    const newDelta = ablate ? null : delta;
    const feature = findSelectedFeature(featLayer, featIndex);
    // feature not found, add it
    if (!feature) {
      // add it to the array with last position
      setSteerLogitFeatures([
        ...steerLogitFeatures,
        { layer: featLayer, index: featIndex, delta: newDelta, start_position: lastPosition, ablate },
      ]);
      return;
    }
    // found the feature, update its multiplier
    setSteerLogitFeatures(
      steerLogitFeatures.map((f) =>
        f.layer === featLayer && f.index === featIndex ? { ...f, delta: newDelta, ablate } : f,
      ),
    );
  }

  const [hoveredFeature, setHoveredFeature] = useState<NeuronWithPartialRelations | undefined>();

  return (
    <div key={node.nodeId} className="flex flex-col gap-y-1">
      <div className="flex flex-row items-center gap-x-1 text-[10px]">
        <div className="mr-1 flex flex-col gap-x-1 pl-0">
          <HoverCard openDelay={0} closeDelay={0}>
            <HoverCardTrigger asChild>
              <Button
                onMouseEnter={() => {
                  // if hoveredFeature has same source and index, don't load it again
                  if (
                    hoveredFeature &&
                    hoveredFeature.layer === sourceId &&
                    hoveredFeature.index === index.toString()
                  ) {
                    return;
                  }

                  // if it's not the same, reset it
                  setHoveredFeature(undefined);

                  fetch(`/api/feature/${modelId}/${sourceId}/${index}`, {
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
                    hoveredFeature.index === index.toString() &&
                    hoveredFeature.activations
                  ) {
                    // if we already loaded this from hover, then just set it and open it
                    setFeatureModalFeature(hoveredFeature);
                  } else {
                    setFeatureModalFeature({
                      modelId,
                      layer: sourceId,
                      index: index.toString(),
                    } as NeuronWithPartialRelations);
                  }
                  setFeatureModalOpen(true);
                }}
                className="group relative mr-0 flex h-7 min-w-[93px] shrink-0 flex-row items-center justify-start whitespace-nowrap rounded-md bg-slate-200/70 px-0 py-[6px] text-[8.5px] font-medium leading-none text-slate-500 shadow-none hover:bg-sky-200 hover:text-sky-700"
              >
                <ArrowUpRightFromSquare className="absolute left-2 h-2.5 w-2.5 text-slate-500 group-hover:text-sky-700" />
                <div className="flex flex-col items-start justify-start gap-y-[2px] pl-[25px] text-left font-mono font-medium">
                  <div className="">LAYER {layer}</div>
                  <div className="">INDEX {index}</div>
                </div>
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
        <div className="flex flex-1 basis-1/2 flex-col gap-y-1">
          <div className="line-clamp-1 flex-1 text-xs" title={label}>
            {label}
          </div>
        </div>
        <div className="mt-0.5 flex w-full basis-1/2 flex-row items-center gap-x-3 px-3 text-sky-700">
          <Slider.Root
            defaultValue={[findSelectedFeature(layer, index)?.delta || 0]}
            min={STEER_STRENGTH_MIN}
            max={STEER_STRENGTH_MAX}
            step={10}
            value={[findSelectedFeature(layer, index)?.delta || 0]}
            onValueChange={(value) => {
              setFeatureDelta(layer, index, value[0]);
            }}
            // disabled={!findSelectedFeature(layer, index)}
            className={`group relative flex h-5 w-full items-center overflow-visible ${
              !findSelectedFeature(layer, index) ? 'opacity-50 hover:opacity-70' : 'cursor-pointer'
            }`}
          >
            <Slider.Track className="relative h-[8px] grow cursor-pointer rounded-full border border-sky-600 bg-sky-600 disabled:border-slate-300 disabled:bg-slate-100 group-hover:bg-sky-700">
              <Slider.Range className="absolute h-full rounded-full" />
              <div className="mx-auto -mt-[9px] h-[24px] w-[1px] bg-sky-600 group-hover:bg-sky-700" />
            </Slider.Track>
            <Slider.Thumb
              onClick={() => {
                if (!findSelectedFeature(layer, index)) {
                  setFeatureDelta(layer, index, 0);
                }
              }}
              className="relative flex h-5 w-10 cursor-pointer items-center justify-center overflow-visible rounded-full border border-sky-700 bg-white text-[10px] font-medium leading-none text-sky-700 shadow disabled:border-slate-300 disabled:bg-slate-100 group-hover:bg-sky-100"
            >
              {!findSelectedFeature(layer, index) ? (
                <MousePointerClick className="h-3.5 w-3.5" />
              ) : findSelectedFeature(layer, index)?.ablate ? (
                <span className="text-[7px] font-bold">ABLATE</span>
              ) : (
                // @ts-ignore
                `${findSelectedFeature(layer, index)?.delta ? (findSelectedFeature(layer, index)?.delta > 0 ? '+' : '') : ''}${findSelectedFeature(layer, index)?.delta || '0'}`
              )}
            </Slider.Thumb>
          </Slider.Root>
          <Button
            variant="ghost"
            size="sm"
            className={`-ml-1.5 flex h-6 w-6 min-w-6 rounded p-0 text-red-700 hover:bg-red-100 ${
              findSelectedFeature(layer, index) ? 'text-red-700 hover:bg-red-100' : 'text-slate-400'
            }`}
            disabled={!findSelectedFeature(layer, index)}
            onClick={() =>
              setSteerLogitFeatures(steerLogitFeatures.filter((f) => !(f.layer === layer && f.index === index)))
            }
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>
      {findSelectedFeature(layer, index) && (
        <div className="mb-2 ml-3 mt-0.5 flex flex-col">
          <div className="mb-[3px] text-[8px] uppercase leading-none text-slate-400">
            SELECT TOKEN TO START STEERING FROM
          </div>
          <div className="flex flex-wrap items-end gap-y-[3px]">
            {selectedGraph?.metadata.prompt_tokens.map((token, i) => (
              // eslint-disable-next-line
              <span
                key={i}
                title="Click to start steering at this position."
                className={`mr-[3px] cursor-pointer rounded border border-transparent px-[3px] py-[1px] font-mono text-[10px] text-slate-800 transition-all ${
                  findSelectedFeature(layer, index) && i >= (findSelectedFeature(layer, index)?.start_position || 0)
                    ? 'bg-sky-600 text-white hover:bg-sky-700'
                    : 'bg-slate-200 hover:border-sky-600'
                } ${BOS_TOKENS.includes(token) ? 'hidden' : ''}`}
                onClick={() => {
                  // update the feature's position
                  setSteerLogitFeatures(
                    steerLogitFeatures.map((f) =>
                      f.layer === layer && f.index === index ? { ...f, start_position: i } : f,
                    ),
                  );
                }}
              >
                {token.toString().replaceAll(' ', '\u00A0').replaceAll('\n', '↵')}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function SteerModal() {
  const { isSteerModalOpen, setIsSteerModalOpen } = useGraphModalContext();
  const { visState, selectedMetadataGraph, selectedGraph, getOverrideClerpForNode } = useGraphContext();
  const [steerResult, setSteerResult] = useState<SteerResponse | undefined>();
  const [isSteering, setIsSteering] = useState(false);
  const [steerLogitFeatures, setSteerLogitFeatures] = useState<SteerLogitFeature[]>([]);
  const [steerTokens, setSteerTokens] = useState(STEER_N_COMPLETION_TOKENS_GRAPH);
  const [temperature, setTemperature] = useState(STEER_TEMPERATURE_GRAPH);
  const [freqPenalty, setFreqPenalty] = useState(STEER_FREQUENCY_PENALTY_GRAPH);
  const [seed, setSeed] = useState(STEER_SEED);
  const [randomSeed, setRandomSeed] = useState(true);
  const [freezeAttention, setFreezeAttention] = useState(STEER_FREEZE_ATTENTION);

  const resetSteerSettings = () => {
    setSteerResult(undefined);
    setIsSteering(false);
    setSteerLogitFeatures([]);
    setSteerTokens(STEER_N_COMPLETION_TOKENS_GRAPH);
    setTemperature(STEER_TEMPERATURE_GRAPH);
    setFreqPenalty(STEER_FREQUENCY_PENALTY_GRAPH);
    setSeed(STEER_SEED);
    setRandomSeed(true);
    setFreezeAttention(STEER_FREEZE_ATTENTION);
  };

  useEffect(() => {
    // reset everything when selected graph changes
    resetSteerSettings();
  }, [selectedGraph]);

  return (
    <Dialog open={isSteerModalOpen} onOpenChange={setIsSteerModalOpen}>
      <DialogContent className="flex h-[90vh] max-h-[90vh] min-h-[90vh] w-full max-w-[92vw] flex-col gap-y-3 overflow-hidden bg-slate-50 pt-4">
        <DialogHeader className="flex w-full flex-col items-center justify-center">
          <DialogTitle className="flex w-full select-none flex-row items-center justify-center text-base text-slate-700">
            Steer/Intervention Mode (Beta)
          </DialogTitle>
          {/* <div className="text-xs text-slate-500">{JSON.stringify(steerLogitFeatures, null, 2)}</div> */}
        </DialogHeader>
        <div className="h-full max-h-full overflow-y-hidden">
          {selectedGraph ? (
            <div className="flex h-full max-h-full w-full flex-row gap-x-4 gap-y-1">
              <div className="flex h-full max-h-full basis-1/2 flex-col gap-y-1 px-0.5 pb-0.5 text-xs">
                <Card className="flex h-full max-h-full w-full flex-col bg-white">
                  <CardHeader className="sticky top-0 z-10 flex w-full flex-row items-center justify-between rounded-t-xl bg-white pb-3 pt-6">
                    <div className="flex flex-col gap-y-1.5">
                      <CardTitle>Features to Steer</CardTitle>
                      <div className="text-xs text-slate-500">
                        Drag sliders to negatively or positively steer the feature. Middle will ablate it.
                      </div>
                    </div>
                    <div className="flex flex-row gap-x-2">
                      <Button
                        onClick={() => {
                          alert(
                            "Oops, this isn't ready yet. Sorry! For now you can only steer features that you have pinned.",
                          );
                        }}
                        size="sm"
                        variant="outline"
                        className="border-slate-300"
                      >
                        + Add Feature
                      </Button>
                      <Button
                        onClick={() => {
                          // eslint-disable-next-line
                          if (confirm('Are you sure you want to clear all features?')) {
                            setSteerLogitFeatures([]);
                          }
                        }}
                        size="sm"
                        variant="outline"
                        disabled={steerLogitFeatures.length === 0}
                        className="group aspect-square border-red-500 px-0 text-red-600 hover:border-red-600 hover:bg-red-100 disabled:border-slate-200 disabled:text-slate-400"
                      >
                        <Trash2 className="h-3.5 w-3.5 group-hover:text-red-700" />
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent className="h-full overflow-y-scroll px-5">
                    {visState.supernodes.length > 0 &&
                      visState.supernodes.map((supernode) => {
                        if (supernode.length === 0) {
                          return null;
                        }
                        return (
                          <div key={supernode.join('-')} className="mb-2 rounded-md bg-slate-50 px-4 py-3 pb-3.5">
                            <div className="mb-2 flex w-full flex-row items-center justify-between gap-x-1.5">
                              <div className="flex flex-row items-end gap-x-1.5 text-[13px]">
                                <div>{supernode[0]}</div>
                                <span className="text-[8px] text-slate-400">SUPERNODE</span>
                              </div>
                              <div className="-mb-1.5 flex basis-1/2 flex-row justify-between pl-[68px] pr-[44px]">
                                <div className="text-[8px] font-medium uppercase leading-none text-slate-400">
                                  ← negative
                                </div>
                                <div className="text-[8px] font-medium uppercase leading-none text-slate-400">
                                  ABLATE
                                </div>
                                <div className="text-[8px] font-medium uppercase leading-none text-slate-400">
                                  positive →
                                </div>
                              </div>
                            </div>
                            <div className="flex flex-col gap-y-1.5 pl-2">
                              {supernode.slice(1).map((id) => {
                                const node = selectedGraph?.nodes.find((n) => n.nodeId === id);
                                if (!node || !nodeTypeHasFeatureDetail(node)) {
                                  return null;
                                }
                                return (
                                  <NodeToSteer
                                    key={id}
                                    node={node}
                                    label={getOverrideClerpForNode(node) || ''}
                                    selectedGraph={selectedGraph}
                                    steerLogitFeatures={steerLogitFeatures}
                                    setSteerLogitFeatures={setSteerLogitFeatures}
                                  />
                                );
                              })}
                            </div>
                          </div>
                        );
                      })}

                    {visState.pinnedIds.length > 0 && (
                      <div className="flex flex-col gap-y-1.5">
                        <div className="rounded-md bg-slate-50 px-4 py-3 pb-3.5">
                          <div className="mb-1.5 flex w-full flex-row items-center justify-between gap-x-1.5">
                            <div className="flex flex-row gap-x-1.5 text-[10px] text-slate-400">
                              NOT IN SUPERNODE GROUP
                            </div>
                            <div className="-mb-1.5 flex basis-1/2 flex-row justify-between pl-[68px] pr-[44px]">
                              <div className="text-[8px] font-medium uppercase leading-none text-slate-400">
                                ← negative
                              </div>
                              <div className="text-[8px] font-medium uppercase leading-none text-slate-400">ABLATE</div>
                              <div className="text-[8px] font-medium uppercase leading-none text-slate-400">
                                positive →
                              </div>
                            </div>
                          </div>
                          <div className="flex flex-col gap-y-1.5 pl-2">
                            {visState.pinnedIds.map((id) => {
                              // find the node and ensure it has a feature detail
                              const node = selectedGraph?.nodes.find((n) => n.nodeId === id);
                              if (!node || !nodeTypeHasFeatureDetail(node)) {
                                return null;
                              }
                              // check if it's in a supernode
                              const supernode = visState.supernodes.find((sn) => sn.includes(id));
                              if (supernode) {
                                return null;
                              }
                              //   // check if any previous pinnedId by index has the same feature
                              //   const previousSameFeatureNode = visState.pinnedIds.slice(0, index).find((id2) => {
                              //     const node2 = selectedGraph?.nodes.find((n) => n.nodeId === id2);
                              //     return node2 && node2.feature === node.feature;
                              //   });
                              //   if (previousSameFeatureNode) {
                              //     return null;
                              //   }
                              return (
                                <NodeToSteer
                                  key={id}
                                  node={node}
                                  label={getOverrideClerpForNode(node) || ''}
                                  selectedGraph={selectedGraph}
                                  steerLogitFeatures={steerLogitFeatures}
                                  setSteerLogitFeatures={setSteerLogitFeatures}
                                />
                              );
                            })}
                          </div>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
              <div className="flex basis-1/2 flex-col gap-y-4 pb-0.5 pr-0.5">
                <Card className="flex w-full flex-col bg-white">
                  <CardHeader className="sticky top-0 z-10 flex w-full flex-row items-center justify-between rounded-t-xl bg-white pb-3 pt-6">
                    <CardTitle>Settings & Steer</CardTitle>
                  </CardHeader>
                  <CardContent className="flex h-full flex-col justify-center px-5">
                    <div className="mt-2 grid w-full grid-cols-3 items-center justify-center gap-x-1 gap-y-1.5">
                      <div className="flex w-full flex-row items-center justify-start gap-x-3">
                        <div className="w-[70px] text-right text-[10px] font-medium uppercase leading-tight text-slate-400">
                          Tokens
                        </div>
                        <input
                          type="number"
                          onChange={(e) => {
                            if (parseInt(e.target.value, 10) < 1) {
                              alert('Tokens must be >= 1');
                              return;
                            }
                            if (parseInt(e.target.value, 10) > STEER_N_COMPLETION_TOKENS_GRAPH_MAX) {
                              alert(
                                `Due to compute constraints, the current allowed max tokens is: ${
                                  STEER_N_COMPLETION_TOKENS_GRAPH_MAX
                                }`,
                              );
                            } else {
                              setSteerTokens(parseInt(e.target.value, 10));
                            }
                          }}
                          className="max-w-[80px] flex-1 rounded-md border-slate-300 py-1 text-center text-xs text-slate-700"
                          value={steerTokens}
                        />
                      </div>
                      <div className="flex w-full flex-row items-center justify-start gap-x-3">
                        <div className="w-[70px] text-right text-[10px] font-medium uppercase leading-tight text-slate-400">
                          Temp
                        </div>
                        <input
                          type="number"
                          onChange={(e) => {
                            if (parseFloat(e.target.value) > STEER_TEMPERATURE_MAX || parseFloat(e.target.value) < 0) {
                              alert(`Temperature must be >= 0 and <= ${STEER_TEMPERATURE_MAX}`);
                            } else {
                              setTemperature(parseFloat(e.target.value));
                            }
                          }}
                          className="max-w-[80px] flex-1 rounded-md border-slate-300 py-1 text-center text-xs text-slate-700"
                          value={temperature}
                        />
                      </div>
                      <div className="flex w-full flex-row items-center justify-start gap-x-3">
                        <div className="w-[70px] text-right text-[10px] font-medium uppercase leading-tight text-slate-400">
                          Freq Penalty
                        </div>
                        <input
                          type="number"
                          onChange={(e) => {
                            if (
                              parseFloat(e.target.value) > STEER_FREQUENCY_PENALTY_MAX ||
                              parseFloat(e.target.value) < STEER_FREQUENCY_PENALTY_MIN
                            ) {
                              alert(
                                `Freq penalty must be >= ${STEER_FREQUENCY_PENALTY_MIN} and <= ${STEER_FREQUENCY_PENALTY_MAX}`,
                              );
                            } else {
                              setFreqPenalty(parseFloat(e.target.value));
                            }
                          }}
                          className="max-w-[80px] flex-1 rounded-md border-slate-300 py-1 text-center text-xs text-slate-700"
                          value={freqPenalty}
                        />
                      </div>
                      <div className="col-span-1 flex w-full flex-row items-center justify-start gap-x-3">
                        <div className="w-[70px] text-right text-[10px] font-medium uppercase leading-tight text-slate-400">
                          Manual Seed
                        </div>
                        <input
                          type="number"
                          disabled={randomSeed}
                          onChange={(e) => {
                            if (parseInt(e.target.value, 10) > 100000000 || parseInt(e.target.value, 10) < -100000000) {
                              alert('Seed must be >= -100000000 and <= 100000000');
                            } else {
                              setSeed(parseInt(e.target.value, 10));
                            }
                          }}
                          className="max-w-[80px] flex-1 rounded-md border-slate-300 py-1 text-center text-xs text-slate-700 disabled:bg-slate-200 disabled:text-slate-400"
                          value={seed}
                        />
                      </div>
                      <div className="col-span-1 flex w-full flex-row items-center justify-start gap-x-3">
                        <div className="w-[70px] text-right text-[10px] font-medium uppercase leading-tight text-slate-400">
                          Random Seed
                        </div>
                        <input
                          onChange={(e) => {
                            setRandomSeed(e.target.checked);
                            setSeed(STEER_SEED);
                          }}
                          type="checkbox"
                          checked={randomSeed}
                          className="h-5 w-5 cursor-pointer rounded border-slate-300 bg-slate-100 py-1 text-center text-xs text-slate-700 checked:bg-slate-500"
                        />
                      </div>
                      <div className="col-span-1 flex w-full flex-row items-center justify-start gap-x-3">
                        <div className="w-[70px] text-right text-[10px] font-medium uppercase leading-tight text-slate-400">
                          Freeze Attn
                        </div>
                        <input
                          onChange={(e) => {
                            setFreezeAttention(e.target.checked);
                          }}
                          type="checkbox"
                          checked={freezeAttention}
                          className="h-5 w-5 cursor-pointer rounded border-slate-300 bg-slate-100 py-1 text-center text-xs text-slate-700 checked:bg-slate-500"
                        />
                      </div>
                    </div>
                    <div className="mt-5 flex w-full flex-row gap-x-2">
                      <Button
                        variant="outline"
                        onClick={() => {
                          resetSteerSettings();
                        }}
                        className="flex flex-1 flex-row gap-x-1.5 self-center text-xs font-bold uppercase text-slate-400"
                      >
                        <ResetIcon className="h-4 w-4" /> Reset
                      </Button>
                      <Button
                        variant="emerald"
                        className="flex flex-1 flex-row gap-x-1.5 self-center text-xs font-bold uppercase"
                        onClick={() => {
                          if (steerLogitFeatures.length === 0) {
                            alert(
                              'Error: You haven\'t chosen any features to ablate or steer.\n\nUnder "Features to Steer", drag a feature\'s slider to the left to negatively steer it, to the right to positively steer it, or to the center to ablate it.',
                            );
                            return;
                          }
                          setIsSteering(true);
                          setSteerResult(undefined);
                          // TODO: remove <bos> hack
                          const requestBody: SteerLogitsRequest = {
                            modelId: selectedGraph?.metadata.scan,
                            prompt: selectedGraph?.metadata.prompt.replaceAll('<bos>', '') || '',
                            features: steerLogitFeatures,
                            nTokens: steerTokens,
                            topK: 5,
                            freezeAttention,
                            temperature,
                            freqPenalty,
                            seed,
                          };

                          // Make the API request
                          fetch('/api/steer-logits', {
                            method: 'POST',
                            headers: {
                              'Content-Type': 'application/json',
                            },
                            body: JSON.stringify(requestBody),
                          })
                            .then((response) => response.json())
                            .then((data: SteerResponse) => {
                              setSteerResult(data);
                              setIsSteering(false);
                            })
                            .catch((error) => {
                              console.error('Error steering logits:', error);
                              setIsSteering(false);
                            });
                        }}
                      >
                        <Joystick className="h-4 w-4" />
                        Steer
                      </Button>
                    </div>
                  </CardContent>
                </Card>
                <Card className="flex h-full w-full flex-col bg-white">
                  <CardHeader className="sticky top-0 z-10 flex w-full flex-row items-center justify-between rounded-t-xl bg-white pb-3 pt-5">
                    <CardTitle>Results</CardTitle>
                  </CardHeader>
                  <CardContent className="flex h-full max-h-full flex-col px-6">
                    <div className="flex-1">
                      <div className="mb-1.5 mt-3 text-xs font-bold uppercase text-slate-400">Prompt</div>
                      <div className="flex flex-wrap items-end gap-x-0 gap-y-[3px]">
                        {selectedMetadataGraph?.promptTokens.map((token, index) => (
                          <span key={index} className="py-[3px] font-mono text-[11px] text-slate-800">
                            {token.toString().replaceAll(' ', '\u00A0').replaceAll('\n', '↵').replaceAll('<bos>', '')}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div className="flex-1">
                      <div className="mb-1.5 mt-3 text-xs font-bold uppercase text-slate-400">Default</div>
                      {steerResult ? (
                        <TokenTooltip logitsByToken={steerResult.DEFAULT_LOGITS_BY_TOKEN} />
                      ) : isSteering ? (
                        <div className="h-10">
                          <LoadingSquare className="h-5 w-5" />
                        </div>
                      ) : (
                        <div className="mt-3 w-full text-xs text-slate-400">
                          Click Steer to generate the default completion.
                        </div>
                      )}
                    </div>
                    <div className="flex-1">
                      <div className="mb-1.5 mt-3 text-xs font-bold uppercase text-slate-400">Steered</div>
                      {steerResult ? (
                        <TokenTooltip logitsByToken={steerResult.STEERED_LOGITS_BY_TOKEN} />
                      ) : isSteering ? (
                        <div className="h-10">
                          <LoadingSquare className="h-5 w-5" />
                        </div>
                      ) : (
                        <div className="mt-3 w-full text-xs text-slate-400">
                          Click Steer to generate the steered completion.
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          ) : null}
        </div>
      </DialogContent>
    </Dialog>
  );
}
