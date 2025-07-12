// TODO: we should use a context for this instead of passing all these functions around

import FeatureDashboard from '@/app/[modelId]/[layer]/[index]/feature-dashboard';
import {
  ANT_MODEL_ID_TO_NEURONPEDIA_MODEL_ID,
  CLTGraphNode,
  getAnthropicFeatureIdFromLayerAndIndex,
  getIndexFromAnthropicFeatureId,
  getLayerFromAnthropicFeatureId,
  MODEL_DIGITS_IN_FEATURE_ID,
  MODEL_TO_SOURCESET_ID,
  nodeTypeHasFeatureDetail,
} from '@/app/[modelId]/graph/utils';
import CustomTooltip from '@/components/custom-tooltip';
import ExplanationsSearcher from '@/components/explanations-searcher';
import { useGraphModalContext } from '@/components/provider/graph-modal-provider';
import { PREFERRED_EXPLANATION_TYPE_NAME, useGraphContext } from '@/components/provider/graph-provider';
import { Button } from '@/components/shadcn/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/shadcn/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/shadcn/dialog';
import { LoadingSquare } from '@/components/svg/loading-square';
import { BOS_TOKENS } from '@/lib/utils/activations';
import { SearchExplanationsType } from '@/lib/utils/general';
import { SteeredPositionIdentifier, SteerLogitFeature, SteerLogitsRequest, SteerResponse } from '@/lib/utils/graph';
import { getLayerNumFromSource } from '@/lib/utils/source';
import {
  STEER_FREEZE_ATTENTION,
  STEER_FREQUENCY_PENALTY_GRAPH,
  STEER_FREQUENCY_PENALTY_MAX,
  STEER_FREQUENCY_PENALTY_MIN,
  STEER_MULTIPLIER_STEP,
  STEER_N_COMPLETION_TOKENS_GRAPH,
  STEER_N_COMPLETION_TOKENS_GRAPH_MAX,
  STEER_SEED,
  STEER_STRENGTH_ADDED_MULTIPLIER_CUSTOM_GRAPH,
  STEER_STRENGTH_ADDED_MULTIPLIER_GRAPH,
  STEER_STRENGTH_ADDED_MULTIPLIER_MAX,
  STEER_STRENGTH_ADDED_MULTIPLIER_MIN,
  STEER_TEMPERATURE_GRAPH,
  STEER_TEMPERATURE_MAX,
} from '@/lib/utils/steer';
import { ExplanationWithPartialRelations, NeuronWithPartialRelations } from '@/prisma/generated/zod';
import { EyeClosedIcon, EyeOpenIcon, QuestionMarkIcon, ResetIcon } from '@radix-ui/react-icons';
import * as Slider from '@radix-ui/react-slider';
import { Joystick, MousePointerClick, Trash2 } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import NodeToSteer from './steer-modal/node-to-steer';
import TokenTooltip from './steer-modal/token-tooltip';

// sometimes comparing the multipliers can be a bit off, so we allow a small tolerance
const DELTA_COMPARISON_TOLERANCE = 0.1;

export default function SteerModal() {
  const { isSteerModalOpen, setIsSteerModalOpen } = useGraphModalContext();
  const { visState, selectedMetadataGraph, selectedGraph, getOverrideClerpForNode } = useGraphContext();
  const [steerResult, setSteerResult] = useState<SteerResponse | undefined>();
  const [isSteering, setIsSteering] = useState(false);
  const [steeredPositions, setSteeredPositions] = useState<SteerLogitFeature[]>([]);
  const [steerTokens, setSteerTokens] = useState(STEER_N_COMPLETION_TOKENS_GRAPH);
  const [temperature, setTemperature] = useState(STEER_TEMPERATURE_GRAPH);
  const [freqPenalty, setFreqPenalty] = useState(STEER_FREQUENCY_PENALTY_GRAPH);
  const [seed, setSeed] = useState(STEER_SEED);
  const [randomSeed, setRandomSeed] = useState(true);
  const [freezeAttention, setFreezeAttention] = useState(STEER_FREEZE_ATTENTION);
  const [showAddFeature, setShowAddFeature] = useState(false);
  const [queuedAddFeature, setQueuedAddFeature] = useState<ExplanationWithPartialRelations | undefined>();
  const [customSteerNodes, setCustomSteerNodes] = useState<CLTGraphNode[]>([]);
  const [modelId, setModelId] = useState<keyof typeof ANT_MODEL_ID_TO_NEURONPEDIA_MODEL_ID>(
    ANT_MODEL_ID_TO_NEURONPEDIA_MODEL_ID[
      selectedGraph?.metadata.scan as keyof typeof ANT_MODEL_ID_TO_NEURONPEDIA_MODEL_ID
    ] as keyof typeof ANT_MODEL_ID_TO_NEURONPEDIA_MODEL_ID,
  );
  const [expandedSupernodeIndexes, setExpandedSupernodeIndexes] = useState<number[]>([]);

  const getFeatureNodeForNodeId = (id: string): CLTGraphNode | null => {
    const node = selectedGraph?.nodes.find((n) => n.nodeId === id);
    if (!node || !nodeTypeHasFeatureDetail(node)) {
      return null;
    }
    return node;
  };

  const makeNodeSteerIdentifier = (node: CLTGraphNode): SteeredPositionIdentifier => ({
    modelId,
    layer: getLayerFromAnthropicFeatureId(modelId, node.feature),
    index: getIndexFromAnthropicFeatureId(modelId, node.feature),
    tokenActivePosition: node.ctx_idx,
  });

  const makeNodeSourceId = (node: CLTGraphNode): string =>
    `${getLayerFromAnthropicFeatureId(modelId, node.feature)}-${MODEL_TO_SOURCESET_ID[selectedGraph?.metadata.scan as keyof typeof MODEL_TO_SOURCESET_ID]}`;

  // checks if a node (identified by layer, index, and token active position) is steered at all
  const isSteered = useCallback(
    (identifier: SteeredPositionIdentifier) =>
      steeredPositions.find(
        (f) =>
          f.layer === identifier.layer &&
          f.index === identifier.index &&
          f.token_active_position === identifier.tokenActivePosition,
      ) !== undefined,
    [steeredPositions],
  );

  const getTopActivationValue = useCallback((node: CLTGraphNode) => {
    if (node.featureDetailNP && node.featureDetailNP?.activations && node.featureDetailNP.activations.length > 0) {
      return node.featureDetailNP.activations[0].maxValue || 0;
    }
    return 0;
  }, []);

  // finds the top activation for this feature and multiplies it by the multipler.
  const getDeltaToAddForMultiplier = useCallback(
    (node: CLTGraphNode, multiplier: number) =>
      // get the top activation
      getTopActivationValue(node) * multiplier,
    [getTopActivationValue],
  );

  const findSteeredPositionByPosition = useCallback(
    (identifier: SteeredPositionIdentifier, position: number) =>
      steeredPositions.find(
        (f) =>
          f.layer === identifier.layer &&
          f.index === identifier.index &&
          f.steer_position === position &&
          f.token_active_position === identifier.tokenActivePosition,
      ),
    [steeredPositions],
  );

  const setSteeredPositionDeltaByPosition = useCallback(
    (identifier: SteeredPositionIdentifier, position: number, delta: number) => {
      const ablate = delta === 0;
      const newDelta = ablate ? null : delta;

      setSteeredPositions((prevSteeredPositions) => {
        const feature = prevSteeredPositions.find(
          (f) =>
            f.layer === identifier.layer &&
            f.index === identifier.index &&
            f.steer_position === position &&
            f.token_active_position === identifier.tokenActivePosition,
        );

        // feature not currently steered, add the steer at the specified position and delta, and also make it steer generated tokens
        if (!feature) {
          return [
            ...prevSteeredPositions,
            {
              layer: identifier.layer,
              index: identifier.index,
              delta: newDelta,
              steer_position: position,
              ablate,
              steer_generated_tokens: false,
              token_active_position: identifier.tokenActivePosition,
            },
          ];
        }

        // feature is currently steered at this position, update the delta
        return prevSteeredPositions.map((f) =>
          f.layer === identifier.layer &&
          f.index === identifier.index &&
          f.steer_position === position &&
          f.token_active_position === identifier.tokenActivePosition
            ? { ...f, delta: newDelta, ablate }
            : f,
        );
      });
    },
    [setSteeredPositions],
  );

  const removeSteeredPosition = useCallback(
    (identifier: SteeredPositionIdentifier) => {
      setSteeredPositions((prevSteeredPositions) =>
        prevSteeredPositions.filter(
          (f) =>
            !(
              f.layer === identifier.layer &&
              f.index === identifier.index &&
              f.token_active_position === identifier.tokenActivePosition
            ),
        ),
      );
    },
    [setSteeredPositions],
  );

  const findSteeredPositionSteerGeneratedTokens = (nodeSteerIdentifier: SteeredPositionIdentifier) =>
    steeredPositions.find(
      (f) =>
        f.layer === nodeSteerIdentifier.layer &&
        f.index === nodeSteerIdentifier.index &&
        f.steer_generated_tokens &&
        f.token_active_position === nodeSteerIdentifier.tokenActivePosition,
    );

  const setSteeredPositionDeltaSteerGeneratedTokens = (
    nodeSteerIdentifier: SteeredPositionIdentifier,
    delta: number,
  ) => {
    const ablate = delta === 0;
    const newDelta = ablate ? null : delta;

    setSteeredPositions((prevSteeredPositions) => {
      const existingFeature = prevSteeredPositions.find(
        (f) =>
          f.layer === nodeSteerIdentifier.layer &&
          f.index === nodeSteerIdentifier.index &&
          f.steer_generated_tokens &&
          f.token_active_position === nodeSteerIdentifier.tokenActivePosition,
      );

      if (existingFeature) {
        return prevSteeredPositions.map((f) =>
          f.layer === nodeSteerIdentifier.layer &&
          f.index === nodeSteerIdentifier.index &&
          f.steer_generated_tokens &&
          f.token_active_position === nodeSteerIdentifier.tokenActivePosition
            ? { ...f, delta: newDelta, ablate }
            : f,
        );
      }
      return [
        ...prevSteeredPositions,
        {
          layer: nodeSteerIdentifier.layer,
          index: nodeSteerIdentifier.index,
          delta: newDelta,
          ablate,
          steer_generated_tokens: true,
          token_active_position: nodeSteerIdentifier.tokenActivePosition,
        },
      ];
    });
  };

  // number is the delta they all have in common, false means not all the same, null = ablate
  const allTokensHaveSameDelta = (nodeSteerIdentifier: SteeredPositionIdentifier): number | false | null => {
    if (steeredPositions.length <= 1 || !selectedGraph) {
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
      const feature = findSteeredPositionByPosition(nodeSteerIdentifier, i);
      if (feature?.delta !== firstDelta) {
        return false;
      }
    }
    // check the "generated" too
    const generatedFeature = findSteeredPositionSteerGeneratedTokens(nodeSteerIdentifier);
    if (generatedFeature?.delta !== firstDelta) {
      return false;
    }
    return firstDelta;
  };

  const setAllTokensDelta = (nodeSteerIdentifier: SteeredPositionIdentifier, delta: number) => {
    if (!selectedGraph) {
      return;
    }

    setSteeredPositions((prevSteeredPositions) => {
      let newSteeredPositionFeatures = prevSteeredPositions;
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
      return newSteeredPositionFeatures;
    });
  };

  const allFeaturesInSupernodeHaveSameMultiplier = (
    supernode: string[],
    position?: number,
    steerGeneratedTokens?: boolean,
  ): number | false | null => {
    let toReturn: number | false | null | undefined;
    // iterate through supernode - ignore first item bc that's the label
    supernode.slice(1).forEach((id) => {
      const node = getFeatureNodeForNodeId(id);
      // ignore all nodes that don't have feature detail (eg mlp error nodes)
      if (!node || !nodeTypeHasFeatureDetail(node)) {
        return;
      }
      const nodeSteerIdentifier = makeNodeSteerIdentifier(node);
      const delta = steerGeneratedTokens
        ? findSteeredPositionSteerGeneratedTokens(nodeSteerIdentifier)?.delta
        : position !== undefined
          ? findSteeredPositionByPosition(nodeSteerIdentifier, position)?.delta
          : allTokensHaveSameDelta(nodeSteerIdentifier);
      if (delta === null) {
        if (toReturn === undefined) {
          toReturn = null;
        } else if (toReturn !== null) {
          toReturn = false;
        }
      } else if (delta === undefined) {
        toReturn = false;
      } else {
        const multiplier = delta ? delta / getTopActivationValue(node) : 0;
        if (toReturn === undefined) {
          toReturn = multiplier;
        } else if (
          typeof toReturn === 'number' &&
          typeof multiplier === 'number' &&
          Math.abs(toReturn - multiplier) > DELTA_COMPARISON_TOLERANCE
        ) {
          toReturn = false;
        }
      }
    });
    if (toReturn === undefined) {
      return false;
    }
    return toReturn;
  };

  const isAtLeastOneNodeInSupernodeSteered = (supernode: string[]) =>
    supernode.some((id) => {
      const node = getFeatureNodeForNodeId(id);
      if (!node) {
        return false;
      }
      return isSteered(makeNodeSteerIdentifier(node));
    });

  const setSteeredPositionMultiplierByPositionForSupernode = (
    supernode: string[],
    position: number,
    delta: number,
    steerGeneratedTokens?: boolean,
  ) => {
    supernode.slice(1).forEach((id) => {
      const node = getFeatureNodeForNodeId(id);
      if (node) {
        if (steerGeneratedTokens) {
          setSteeredPositionDeltaSteerGeneratedTokens(
            makeNodeSteerIdentifier(node),
            getDeltaToAddForMultiplier(node, delta),
          );
        } else {
          setSteeredPositionDeltaByPosition(
            makeNodeSteerIdentifier(node),
            position,
            getDeltaToAddForMultiplier(node, delta),
          );
        }
      }
    });
  };

  const findSteeredNodesInSupernodeAtPosition = (
    supernode: string[],
    position: number,
    steerGeneratedTokens?: boolean,
  ) => {
    // find all nodes in steeredPositions that are in this supernode and have steer_position === position
    const matchedNodes = steeredPositions.filter((f) => {
      // check if the steered position is in this supernode
      const nodesInSupernode = supernode
        .slice(1)
        .map((id) => getFeatureNodeForNodeId(id))
        .filter((node) => node !== null);
      if (nodesInSupernode.length === 0) {
        return false;
      }
      if (steerGeneratedTokens) {
        return nodesInSupernode.some(
          (node) =>
            f.layer === getLayerFromAnthropicFeatureId(modelId, node.feature) &&
            f.index === getIndexFromAnthropicFeatureId(modelId, node.feature) &&
            f.steer_generated_tokens &&
            f.token_active_position === node.ctx_idx,
        );
      }
      return nodesInSupernode.some(
        (node) =>
          f.layer === getLayerFromAnthropicFeatureId(modelId, node.feature) &&
          f.index === getIndexFromAnthropicFeatureId(modelId, node.feature) &&
          f.steer_position === position &&
          f.token_active_position === node.ctx_idx,
      );
    });
    return matchedNodes;
  };

  const lastSupernodeIsSteered = (supernode: string[]) => {
    const lastNode = supernode[supernode.length - 1];
    const node = getFeatureNodeForNodeId(lastNode);
    if (!node) {
      return false;
    }
    return isSteered(makeNodeSteerIdentifier(node));
  };

  const resetSteerSettings = () => {
    setSteerResult(undefined);
    setIsSteering(false);
    setSteeredPositions(() => []);
    setCustomSteerNodes(() => []);
    setSteerTokens(STEER_N_COMPLETION_TOKENS_GRAPH);
    setTemperature(STEER_TEMPERATURE_GRAPH);
    setFreqPenalty(STEER_FREQUENCY_PENALTY_GRAPH);
    setSeed(STEER_SEED);
    setModelId(
      ANT_MODEL_ID_TO_NEURONPEDIA_MODEL_ID[
        selectedGraph?.metadata.scan as keyof typeof ANT_MODEL_ID_TO_NEURONPEDIA_MODEL_ID
      ] as keyof typeof ANT_MODEL_ID_TO_NEURONPEDIA_MODEL_ID,
    );
    setRandomSeed(true);
    setExpandedSupernodeIndexes([]);
    setFreezeAttention(STEER_FREEZE_ATTENTION);
  };

  useEffect(() => {
    // reset everything when selected graph changes
    resetSteerSettings();
  }, [selectedGraph]);

  return (
    <Dialog open={isSteerModalOpen} onOpenChange={setIsSteerModalOpen}>
      <DialogContent className="flex h-[90vh] max-h-[90vh] min-h-[90vh] w-full max-w-[95vw] flex-col gap-y-3 overflow-hidden bg-slate-50 pt-4">
        <DialogHeader className="flex w-full flex-col items-center justify-center">
          <DialogTitle className="flex w-full select-none flex-row items-center justify-center text-base text-slate-700">
            Steer/Intervention Mode (Beta)
          </DialogTitle>
          {/* <div className="text-xs text-slate-500">{JSON.stringify(steeredPositions, null, 2)}</div> */}
        </DialogHeader>
        {showAddFeature && (
          <div className="absolute left-0 top-0 z-10 flex h-full w-full flex-row items-center justify-center gap-y-1.5 px-8 py-3">
            <div className="absolute left-0 top-0 h-full w-full bg-slate-400/20 backdrop-blur-lg" />
            <div className="relative flex h-full max-h-[650px] w-full max-w-[1075px] flex-col overflow-hidden rounded-md border border-slate-200 bg-white p-3 shadow-md">
              <Button
                variant="outline"
                className="absolute left-3 top-3 bg-slate-100 text-[11px] font-normal uppercase text-slate-500 hover:bg-slate-200"
                onClick={() => setShowAddFeature(false)}
                size="sm"
              >
                Cancel
              </Button>
              <div className="mb-0 mt-0.5 text-center text-base font-bold text-slate-600">
                {queuedAddFeature ? 'Select Feature Active Position' : 'Search for Features to Steer'}
              </div>

              {queuedAddFeature ? (
                <div className="flex h-full max-h-full flex-1 flex-col items-center justify-center px-5 py-3 text-center text-base font-bold text-slate-500">
                  <div className="mb-3 text-xs font-medium leading-loose">
                    To finish adding this feature for steering, click the token where this feature is active in the
                    prompt.
                    <br />
                    This feature will be steered positively at 1x on the selected position.
                  </div>
                  <div className="mb-8 flex flex-wrap items-end justify-center">
                    {selectedMetadataGraph?.promptTokens.map((token, index) => (
                      <Button
                        key={index}
                        variant="default"
                        size="sm"
                        onClick={() => {
                          setQueuedAddFeature(undefined);
                          setShowAddFeature(false);
                          if (queuedAddFeature.neuron?.explanations) {
                            queuedAddFeature.neuron.explanations = [
                              {
                                typeName: PREFERRED_EXPLANATION_TYPE_NAME,
                                description: queuedAddFeature.description,
                              },
                            ];
                          }
                          // make a fake CLTGraphNode so we can steer it
                          const node: CLTGraphNode = {
                            nodeId: `${queuedAddFeature.neuron?.modelId}-${queuedAddFeature.neuron?.layer}-${queuedAddFeature.neuron?.index}`,
                            feature: getAnthropicFeatureIdFromLayerAndIndex(
                              queuedAddFeature.neuron?.modelId as keyof typeof MODEL_DIGITS_IN_FEATURE_ID,
                              getLayerNumFromSource(queuedAddFeature.neuron?.layer || ''),
                              parseInt(queuedAddFeature.neuron?.index || '0', 10),
                            ),
                            layer: getLayerNumFromSource(queuedAddFeature.neuron?.layer || '').toString(),
                            ctx_idx: index,
                            feature_type: 'cross layer transcoder',
                            featureDetailNP: queuedAddFeature.neuron as NeuronWithPartialRelations,
                            activation: 0,
                            node_id: `${queuedAddFeature.neuron?.modelId}-${queuedAddFeature.neuron?.layer}-${queuedAddFeature.neuron?.index}`,
                            token_prob: 0,
                            is_target_logit: true,
                            run_idx: 0,
                            reverse_ctx_idx: 0,
                            jsNodeId: `${queuedAddFeature.neuron?.modelId}-${queuedAddFeature.neuron?.layer}-${queuedAddFeature.neuron?.index}`,
                            clerp: queuedAddFeature.description,
                          };
                          setCustomSteerNodes((prevCustomSteerNodes) => [...prevCustomSteerNodes, node]);
                          setSteeredPositions((prevSteeredPositions) => [
                            ...prevSteeredPositions,
                            {
                              ...makeNodeSteerIdentifier(node),
                              delta: getDeltaToAddForMultiplier(node, STEER_STRENGTH_ADDED_MULTIPLIER_CUSTOM_GRAPH),
                              ablate: false,
                              steer_position: node.ctx_idx,
                              steer_generated_tokens: false,
                              token_active_position: node.ctx_idx,
                            },
                          ]);
                        }}
                        className={`mx-0.5 bg-slate-200 px-1 font-mono text-[11px] text-base font-medium text-slate-600 shadow-none hover:bg-sky-200 hover:text-sky-700 ${
                          BOS_TOKENS.includes(token) ? 'hidden' : ''
                        }`}
                      >
                        {token.toString().replaceAll(' ', '\u00A0').replaceAll('\n', '↵')}
                      </Button>
                    ))}
                  </div>
                  <div className="mb-2 flex max-w-screen-sm flex-col text-center text-[10px] font-medium text-slate-400">
                    <div className="mb-0">SELECTED FEATURE</div>
                    <div className="mb-1 text-xs text-slate-500">{queuedAddFeature.description}</div>
                    <FeatureDashboard
                      key={`${queuedAddFeature?.neuron?.modelId}-${queuedAddFeature?.neuron?.layer}-${queuedAddFeature?.neuron?.index}`}
                      initialNeuron={queuedAddFeature.neuron as NeuronWithPartialRelations}
                      embed
                      forceMiniStats
                    />
                  </div>
                </div>
              ) : (
                <div className="flex h-full max-h-full flex-1 flex-col">
                  <ExplanationsSearcher
                    initialModelId={
                      ANT_MODEL_ID_TO_NEURONPEDIA_MODEL_ID[
                        selectedGraph?.metadata.scan as keyof typeof ANT_MODEL_ID_TO_NEURONPEDIA_MODEL_ID
                      ] || ''
                    }
                    defaultTab={SearchExplanationsType.BY_SOURCE}
                    initialSourceSetName={
                      MODEL_TO_SOURCESET_ID[selectedGraph?.metadata.scan as keyof typeof MODEL_TO_SOURCESET_ID] || ''
                    }
                    showTabs={false}
                    showModelSelector={false}
                    allowSourceSetChange={false}
                    isSteerSearch
                    allowSteerSearchFullHeight
                    onClickResultCallback={(result) => {
                      setQueuedAddFeature(result);
                    }}
                    neverChangePageOnSearch
                  />
                </div>
              )}
            </div>
          </div>
        )}
        <div className="relative h-full max-h-full overflow-y-hidden">
          {selectedGraph ? (
            <div className="grid h-full max-h-full w-full grid-cols-2 gap-x-4 gap-y-1">
              <div className="flex h-full max-h-full min-h-0 flex-1 flex-col gap-y-1 px-0.5 pb-0.5 text-xs">
                <Card className="flex h-full max-h-full w-full flex-col bg-white">
                  <CardHeader className="sticky top-0 flex w-full flex-row items-center justify-between gap-x-5 rounded-t-xl bg-white pb-3 pt-6">
                    <div className="flex flex-col gap-y-1.5">
                      <CardTitle>Features to Steer</CardTitle>
                      <div className="text-xs text-slate-500">
                        Click Steer on a feature. By default, this negatively steers the feature at the position where
                        it was active. You can also drag the sliders to steer the feature at specific positions. Middle
                        will ablate that feature.
                      </div>
                    </div>
                    <div className="flex flex-row gap-x-2">
                      <Button
                        onClick={() => {
                          setShowAddFeature(!showAddFeature);
                        }}
                        size="sm"
                        variant="outline"
                        className="w-28 border-slate-300"
                      >
                        {showAddFeature ? 'Close' : '+ Add Feature'}
                      </Button>
                      <Button
                        onClick={() => {
                          // eslint-disable-next-line
                          if (confirm('Are you sure you want to clear all steering?')) {
                            setSteeredPositions(() => []);
                          }
                        }}
                        size="sm"
                        variant="outline"
                        disabled={steeredPositions.length === 0 || showAddFeature}
                        className="group aspect-square border-red-500 px-0 text-red-600 hover:border-red-600 hover:bg-red-100 disabled:border-slate-200 disabled:text-slate-400"
                      >
                        <Trash2 className="h-3.5 w-3.5 group-hover:text-red-700" />
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent className="relative h-full overflow-y-scroll px-5">
                    {customSteerNodes.length > 0 && (
                      <div className="mb-2 flex flex-col gap-y-1.5">
                        <div className="rounded-md bg-slate-50 px-4 py-3 pb-3.5">
                          <div className="mb-1.5 flex w-full flex-row items-center justify-between gap-x-1.5">
                            <div className="flex flex-row gap-x-1.5 text-[10px] text-slate-400">ADDED FEATURES</div>
                          </div>
                          <div className="flex flex-col gap-y-1.5">
                            {customSteerNodes.map((customNode) => (
                              <div
                                key={customNode.nodeId}
                                className="flex flex-row items-center justify-between gap-x-1.5"
                              >
                                <NodeToSteer
                                  nodeSteerIdentifier={{
                                    modelId,
                                    layer: getLayerFromAnthropicFeatureId(modelId, customNode.feature),
                                    index: getIndexFromAnthropicFeatureId(modelId, customNode.feature),
                                    tokenActivePosition: customNode.ctx_idx,
                                  }}
                                  isCustomSteerNode
                                  sourceId={`${getLayerFromAnthropicFeatureId(modelId, customNode.feature)}-${MODEL_TO_SOURCESET_ID[selectedGraph?.metadata.scan as keyof typeof MODEL_TO_SOURCESET_ID]}`}
                                  node={customNode}
                                  label={getOverrideClerpForNode(customNode) || ''}
                                  selectedGraph={selectedGraph}
                                  steeredPositions={steeredPositions}
                                  setSteeredPositions={setSteeredPositions}
                                  isSteered={isSteered}
                                  findSteeredPositionByPosition={findSteeredPositionByPosition}
                                  removeSteeredPosition={removeSteeredPosition}
                                  setSteeredPositionDeltaByPosition={setSteeredPositionDeltaByPosition}
                                  getDeltaToAddForMultiplier={getDeltaToAddForMultiplier}
                                  getTopActivationValue={getTopActivationValue}
                                  allTokensHaveSameDelta={allTokensHaveSameDelta}
                                  findSteeredPositionSteerGeneratedTokens={findSteeredPositionSteerGeneratedTokens}
                                  setSteeredPositionDeltaSteerGeneratedTokens={
                                    setSteeredPositionDeltaSteerGeneratedTokens
                                  }
                                  setAllTokensDelta={setAllTokensDelta}
                                />
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className={`flex h-6 w-6 min-w-6 rounded p-0 text-red-700 hover:bg-red-100 ${
                                    // if this is being steered, hide the trash can
                                    steeredPositions.some(
                                      (f) =>
                                        f.layer ===
                                          getLayerFromAnthropicFeatureId(
                                            ANT_MODEL_ID_TO_NEURONPEDIA_MODEL_ID[
                                              selectedGraph?.metadata
                                                .scan as keyof typeof ANT_MODEL_ID_TO_NEURONPEDIA_MODEL_ID
                                            ] as keyof typeof ANT_MODEL_ID_TO_NEURONPEDIA_MODEL_ID,
                                            customNode.feature,
                                          ) &&
                                        f.index ===
                                          getIndexFromAnthropicFeatureId(
                                            ANT_MODEL_ID_TO_NEURONPEDIA_MODEL_ID[
                                              selectedGraph?.metadata
                                                .scan as keyof typeof ANT_MODEL_ID_TO_NEURONPEDIA_MODEL_ID
                                            ] as keyof typeof ANT_MODEL_ID_TO_NEURONPEDIA_MODEL_ID,
                                            customNode.feature,
                                          ) &&
                                        f.token_active_position === customNode.ctx_idx,
                                    )
                                      ? 'hidden'
                                      : ''
                                  }`}
                                  onClick={() => {
                                    setCustomSteerNodes((prevCustomSteerNodes) =>
                                      prevCustomSteerNodes.filter((node) => node.nodeId !== customNode.nodeId),
                                    );
                                  }}
                                >
                                  <Trash2 className="h-3.5 w-3.5" />
                                </Button>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}

                    {visState.supernodes.length > 0 &&
                      visState.supernodes.map((supernode, supernodeIndex) => {
                        if (supernode.length === 0) {
                          return null;
                        }
                        return (
                          <div
                            key={supernode.join('-')}
                            className="relative mb-2 rounded-md bg-slate-50 px-4 py-3 pb-3.5"
                          >
                            <div className="mb-2.5 flex w-full flex-row items-center justify-between gap-x-1.5">
                              <div className="flex flex-row items-end gap-x-1.5 text-[13px]">
                                <div>{supernode[0]}</div>
                                <span className="text-[8px] text-slate-400">SUPERNODE</span>
                              </div>
                            </div>
                            {isAtLeastOneNodeInSupernodeSteered(supernode) && lastSupernodeIsSteered(supernode) ? (
                              <div className="absolute left-7 top-11 z-0 h-[calc(100%_-_246px)] w-[1px] bg-sky-700" />
                            ) : (
                              <div className="absolute left-7 top-11 z-0 h-[calc(100%_-_73px)] w-[1px] bg-sky-700" />
                            )}

                            <div className="mb-2 flex flex-row items-center justify-start gap-x-2">
                              <Button
                                onClick={() => {
                                  if (isAtLeastOneNodeInSupernodeSteered(supernode)) {
                                    // remove all nodes in supernode
                                    supernode.forEach((id) => {
                                      const node = getFeatureNodeForNodeId(id);
                                      if (node) {
                                        removeSteeredPosition(makeNodeSteerIdentifier(node));
                                      }
                                    });
                                    // we did unsteer, so we hide the supernode controls
                                    setExpandedSupernodeIndexes(
                                      expandedSupernodeIndexes.filter((i) => i !== supernodeIndex),
                                    );
                                  } else {
                                    // steer all nodes in supernode
                                    supernode.forEach((id) => {
                                      const node = getFeatureNodeForNodeId(id);
                                      if (node) {
                                        setSteeredPositionDeltaByPosition(
                                          makeNodeSteerIdentifier(node),
                                          node.ctx_idx,
                                          getDeltaToAddForMultiplier(node, STEER_STRENGTH_ADDED_MULTIPLIER_GRAPH),
                                        );
                                      }
                                    });
                                    // we did expand, so we show the supernode controls
                                    setExpandedSupernodeIndexes([...expandedSupernodeIndexes, supernodeIndex]);
                                  }
                                }}
                                className={`relative z-10 h-6 w-24 rounded-full border text-[9px] font-medium uppercase ${
                                  isAtLeastOneNodeInSupernodeSteered(supernode)
                                    ? 'border-red-600 bg-red-50 text-red-600 hover:bg-red-100'
                                    : 'border-sky-700 bg-white text-sky-800 hover:bg-sky-200'
                                }`}
                              >
                                {isAtLeastOneNodeInSupernodeSteered(supernode) ? 'Unsteer All' : 'Steer All'}
                              </Button>
                              {isAtLeastOneNodeInSupernodeSteered(supernode) && (
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => {
                                    if (expandedSupernodeIndexes.includes(supernodeIndex)) {
                                      setExpandedSupernodeIndexes(
                                        expandedSupernodeIndexes.filter((i) => i !== supernodeIndex),
                                      );
                                    } else {
                                      setExpandedSupernodeIndexes([...expandedSupernodeIndexes, supernodeIndex]);
                                    }
                                  }}
                                  className="flex h-6 w-[120px] flex-row items-center justify-center gap-x-1.5 rounded-full border-transparent bg-slate-200 px-0 py-0 text-[8.5px] uppercase text-slate-500 hover:border-slate-300 hover:bg-slate-50 hover:text-slate-600"
                                >
                                  {expandedSupernodeIndexes.includes(supernodeIndex) ? (
                                    <>
                                      <EyeOpenIcon className="h-3 w-3" />
                                      Group Controls
                                    </>
                                  ) : (
                                    <>
                                      <EyeClosedIcon className="h-3 w-3" />
                                      Group Controls
                                    </>
                                  )}
                                </Button>
                              )}
                            </div>

                            {/* == start == steer the entire supernode ==  TODO: reduce duplicated code with nodeToSteer == */}
                            {isAtLeastOneNodeInSupernodeSteered(supernode) &&
                              expandedSupernodeIndexes.includes(supernodeIndex) && (
                                <div className="flex max-w-full flex-1 flex-row items-start gap-x-1.5 pb-2 pl-6">
                                  <div className="flex min-w-11 flex-col items-center justify-center gap-y-1 rounded bg-slate-200/50 py-2">
                                    <Slider.Root
                                      orientation="vertical"
                                      defaultValue={[0]}
                                      min={STEER_STRENGTH_ADDED_MULTIPLIER_MIN}
                                      max={STEER_STRENGTH_ADDED_MULTIPLIER_MAX}
                                      step={STEER_MULTIPLIER_STEP}
                                      value={(() => {
                                        const newMultiplier = allFeaturesInSupernodeHaveSameMultiplier(supernode);
                                        const multiplier = newMultiplier || false;
                                        if (multiplier === null || multiplier === false) {
                                          return [0];
                                        }
                                        return [Number(multiplier.toFixed(1)) || 0];
                                      })()}
                                      onValueChange={(value) => {
                                        // Apply the steering to all nodes in the supernode
                                        supernode.slice(1).forEach((id) => {
                                          const node = getFeatureNodeForNodeId(id);
                                          if (node) {
                                            setAllTokensDelta(
                                              makeNodeSteerIdentifier(node),
                                              getDeltaToAddForMultiplier(node, value[0]),
                                            );
                                          }
                                        });
                                      }}
                                      className={`group relative flex h-24 w-2 items-center justify-center overflow-visible ${
                                        allFeaturesInSupernodeHaveSameMultiplier(supernode) === false ||
                                        allFeaturesInSupernodeHaveSameMultiplier(supernode) === 0
                                          ? 'opacity-50 hover:opacity-70'
                                          : 'cursor-pointer'
                                      }`}
                                    >
                                      <Slider.Track className="relative h-full w-[4px] grow cursor-pointer rounded-full border border-sky-600 bg-sky-600 disabled:border-slate-300 disabled:bg-slate-100 group-hover:bg-sky-700">
                                        <Slider.Range className="absolute h-full rounded-full" />
                                      </Slider.Track>
                                      <Slider.Thumb
                                        onClick={() => {
                                          if (!allFeaturesInSupernodeHaveSameMultiplier(supernode)) {
                                            supernode.forEach((id) => {
                                              const node = getFeatureNodeForNodeId(id);
                                              if (node) {
                                                setAllTokensDelta(makeNodeSteerIdentifier(node), 0);
                                              }
                                            });
                                          }
                                        }}
                                        className="relative flex h-5 w-9 cursor-pointer select-none items-center justify-center overflow-visible rounded-full border border-sky-700 bg-white text-[10px] font-medium leading-none text-sky-700 shadow disabled:border-slate-300 disabled:bg-slate-100 group-hover:bg-sky-100"
                                      >
                                        {allFeaturesInSupernodeHaveSameMultiplier(supernode) === false ||
                                        allFeaturesInSupernodeHaveSameMultiplier(supernode) === 0 ? (
                                          <MousePointerClick className="h-3.5 w-3.5" />
                                        ) : allFeaturesInSupernodeHaveSameMultiplier(supernode) === null ? (
                                          <span className="text-[7px] font-bold">ABLATE</span>
                                        ) : (
                                          // @ts-ignore
                                          `${allFeaturesInSupernodeHaveSameMultiplier(supernode) ? (allFeaturesInSupernodeHaveSameMultiplier(supernode) > 0 ? '+' : '') : ''}${(allFeaturesInSupernodeHaveSameMultiplier(supernode) || 0).toFixed(1) || '0'}×`
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
                                        key={`${token}-${i}`}
                                        // ref={node.ctx_idx === i ? scrollIntoViewRef : undefined}
                                        className={`mx-1.5 min-w-fit flex-col items-center justify-center gap-y-1 ${
                                          BOS_TOKENS.includes(token) ? 'hidden' : 'flex'
                                        }`}
                                      >
                                        <Slider.Root
                                          orientation="vertical"
                                          defaultValue={[0]}
                                          min={STEER_STRENGTH_ADDED_MULTIPLIER_MIN}
                                          max={STEER_STRENGTH_ADDED_MULTIPLIER_MAX}
                                          step={STEER_MULTIPLIER_STEP}
                                          value={[allFeaturesInSupernodeHaveSameMultiplier(supernode, i) || 0]}
                                          onValueChange={(value) => {
                                            setSteeredPositionMultiplierByPositionForSupernode(supernode, i, value[0]);
                                          }}
                                          className={`group relative flex h-24 w-2 items-center justify-center overflow-visible ${
                                            allFeaturesInSupernodeHaveSameMultiplier(supernode, i) === false ||
                                            allFeaturesInSupernodeHaveSameMultiplier(supernode, i) === 0
                                              ? 'opacity-50 hover:opacity-70'
                                              : 'cursor-pointer'
                                          }`}
                                        >
                                          <Slider.Track className="relative h-full w-[4px] grow cursor-pointer rounded-full border border-sky-600 bg-sky-600 disabled:border-slate-300 disabled:bg-slate-100 group-hover:bg-sky-700">
                                            <Slider.Range className="absolute h-full rounded-full" />
                                          </Slider.Track>
                                          <Slider.Thumb
                                            onClick={() => {
                                              if (!allFeaturesInSupernodeHaveSameMultiplier(supernode, i)) {
                                                supernode.slice(1).forEach((id) => {
                                                  const node = getFeatureNodeForNodeId(id);
                                                  if (node) {
                                                    setSteeredPositionDeltaByPosition(
                                                      makeNodeSteerIdentifier(node),
                                                      i,
                                                      0,
                                                    );
                                                  }
                                                });
                                              }
                                            }}
                                            className="relative flex h-5 w-9 cursor-pointer select-none items-center justify-center overflow-visible rounded-full border border-sky-700 bg-white text-[10px] font-medium leading-none text-sky-700 shadow disabled:border-slate-300 disabled:bg-slate-100 group-hover:bg-sky-100"
                                          >
                                            {allFeaturesInSupernodeHaveSameMultiplier(supernode, i) === false ||
                                            allFeaturesInSupernodeHaveSameMultiplier(supernode, i) === 0 ? (
                                              <MousePointerClick className="h-3.5 w-3.5" />
                                            ) : allFeaturesInSupernodeHaveSameMultiplier(supernode, i) === null ? (
                                              <span className="text-[7px] font-bold">ABLATE</span>
                                            ) : (
                                              // @ts-ignore
                                              `${allFeaturesInSupernodeHaveSameMultiplier(supernode, i) ? (allFeaturesInSupernodeHaveSameMultiplier(supernode, i) > 0 ? '+' : '') : ''}${(allFeaturesInSupernodeHaveSameMultiplier(supernode, i) || 0).toFixed(1) || '0'}×`
                                            )}
                                          </Slider.Thumb>
                                        </Slider.Root>
                                        <div className="mt-0.5 h-5 rounded bg-slate-200 px-1 py-0.5 font-mono text-[8.5px] text-slate-700">
                                          {token.toString().replaceAll(' ', '\u00A0').replaceAll('\n', '↵')}
                                        </div>
                                        <Button
                                          variant="ghost"
                                          size="sm"
                                          className={`flex h-6 w-6 min-w-6 rounded p-0 ${
                                            allFeaturesInSupernodeHaveSameMultiplier(supernode, i) === false ||
                                            allFeaturesInSupernodeHaveSameMultiplier(supernode, i) === 0
                                              ? 'text-slate-400'
                                              : 'text-red-700 hover:bg-red-100'
                                          }`}
                                          disabled={
                                            allFeaturesInSupernodeHaveSameMultiplier(supernode, i) === false ||
                                            allFeaturesInSupernodeHaveSameMultiplier(supernode, i) === 0
                                          }
                                          onClick={() => {
                                            const matchedNodes = findSteeredNodesInSupernodeAtPosition(supernode, i);
                                            const newSteeredPositions = steeredPositions.filter((f) => {
                                              if (
                                                matchedNodes.some(
                                                  (m) =>
                                                    i === f.steer_position &&
                                                    m.layer === f.layer &&
                                                    m.index === f.index &&
                                                    m.token_active_position === f.token_active_position,
                                                )
                                              ) {
                                                return false;
                                              }
                                              return true;
                                            });
                                            setSteeredPositions(newSteeredPositions);
                                          }}
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
                                      step={STEER_MULTIPLIER_STEP}
                                      value={[
                                        allFeaturesInSupernodeHaveSameMultiplier(supernode, undefined, true) || 0,
                                      ]}
                                      onValueChange={(value) => {
                                        // position (second arg) is ignored here
                                        setSteeredPositionMultiplierByPositionForSupernode(
                                          supernode,
                                          0,
                                          value[0],
                                          true,
                                        );
                                      }}
                                      className={`group relative flex h-24 w-2 items-center justify-center overflow-visible ${
                                        allFeaturesInSupernodeHaveSameMultiplier(supernode, undefined, true) ===
                                          false ||
                                        allFeaturesInSupernodeHaveSameMultiplier(supernode, undefined, true) === 0
                                          ? 'opacity-50 hover:opacity-70'
                                          : 'cursor-pointer'
                                      }`}
                                    >
                                      <Slider.Track className="relative h-full w-[4px] grow cursor-pointer rounded-full border border-sky-600 bg-sky-600 disabled:border-slate-300 disabled:bg-slate-100 group-hover:bg-sky-700">
                                        <Slider.Range className="absolute h-full rounded-full" />
                                      </Slider.Track>
                                      <Slider.Thumb
                                        onClick={() => {
                                          if (!allFeaturesInSupernodeHaveSameMultiplier(supernode, undefined, true)) {
                                            supernode.slice(1).forEach((id) => {
                                              const node = getFeatureNodeForNodeId(id);
                                              if (node) {
                                                setSteeredPositionDeltaSteerGeneratedTokens(
                                                  makeNodeSteerIdentifier(node),
                                                  0,
                                                );
                                              }
                                            });
                                          }
                                        }}
                                        className="relative flex h-5 w-9 cursor-pointer select-none items-center justify-center overflow-visible rounded-full border border-sky-700 bg-white text-[10px] font-medium leading-none text-sky-700 shadow disabled:border-slate-300 disabled:bg-slate-100 group-hover:bg-sky-100"
                                      >
                                        {allFeaturesInSupernodeHaveSameMultiplier(supernode, undefined, true) ===
                                          false ||
                                        allFeaturesInSupernodeHaveSameMultiplier(supernode, undefined, true) === 0 ? (
                                          <MousePointerClick className="h-3.5 w-3.5" />
                                        ) : allFeaturesInSupernodeHaveSameMultiplier(supernode, undefined, true) ===
                                          null ? (
                                          <span className="text-[7px] font-bold">ABLATE</span>
                                        ) : (
                                          // @ts-ignore
                                          `${allFeaturesInSupernodeHaveSameMultiplier(supernode, undefined, true) ? (allFeaturesInSupernodeHaveSameMultiplier(supernode, undefined, true) > 0 ? '+' : '') : ''}${(allFeaturesInSupernodeHaveSameMultiplier(supernode, undefined, true) || 0).toFixed(1) || '0'}×`
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
                                        allFeaturesInSupernodeHaveSameMultiplier(supernode, undefined, true) ===
                                          false ||
                                        allFeaturesInSupernodeHaveSameMultiplier(supernode, undefined, true) === 0
                                          ? 'text-slate-400'
                                          : 'text-red-700 hover:bg-red-100'
                                      }`}
                                      disabled={
                                        allFeaturesInSupernodeHaveSameMultiplier(supernode, undefined, true) ===
                                          false ||
                                        allFeaturesInSupernodeHaveSameMultiplier(supernode, undefined, true) === 0
                                      }
                                      onClick={() => {
                                        const matchedNodes = findSteeredNodesInSupernodeAtPosition(supernode, 0, true);
                                        const newSteeredPositions = steeredPositions.filter((f) => {
                                          if (
                                            matchedNodes.some(
                                              (m) =>
                                                f.steer_generated_tokens &&
                                                m.layer === f.layer &&
                                                m.index === f.index &&
                                                m.token_active_position === f.token_active_position,
                                            )
                                          ) {
                                            return false;
                                          }
                                          return true;
                                        });
                                        setSteeredPositions(newSteeredPositions);
                                      }}
                                    >
                                      <Trash2 className="h-3.5 w-3.5" />
                                    </Button>
                                  </div>
                                </div>
                              )}
                            {/* == end == steer the entire supernode == */}

                            <div className="flex flex-col gap-y-1.5">
                              {supernode.slice(1).map((id) => {
                                const node = getFeatureNodeForNodeId(id);
                                if (!node) {
                                  return null;
                                }
                                return (
                                  <NodeToSteer
                                    key={id}
                                    nodeSteerIdentifier={makeNodeSteerIdentifier(node)}
                                    sourceId={makeNodeSourceId(node)}
                                    node={node}
                                    label={getOverrideClerpForNode(node) || ''}
                                    selectedGraph={selectedGraph}
                                    steeredPositions={steeredPositions}
                                    setSteeredPositions={setSteeredPositions}
                                    isSteered={isSteered}
                                    findSteeredPositionByPosition={findSteeredPositionByPosition}
                                    removeSteeredPosition={removeSteeredPosition}
                                    setSteeredPositionDeltaByPosition={setSteeredPositionDeltaByPosition}
                                    getDeltaToAddForMultiplier={getDeltaToAddForMultiplier}
                                    getTopActivationValue={getTopActivationValue}
                                    allTokensHaveSameDelta={allTokensHaveSameDelta}
                                    findSteeredPositionSteerGeneratedTokens={findSteeredPositionSteerGeneratedTokens}
                                    setSteeredPositionDeltaSteerGeneratedTokens={
                                      setSteeredPositionDeltaSteerGeneratedTokens
                                    }
                                    setAllTokensDelta={setAllTokensDelta}
                                    isInSupernode
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
                          </div>
                          <div className="flex flex-col gap-y-1.5">
                            {visState.pinnedIds.map((id) => {
                              // find the node and ensure it has a feature detail
                              const node = getFeatureNodeForNodeId(id);
                              if (!node) {
                                return null;
                              }
                              // check if it's in a supernode
                              const supernode = visState.supernodes.find((sn) => sn.includes(id));
                              if (supernode) {
                                return null;
                              }
                              return (
                                <NodeToSteer
                                  key={id}
                                  nodeSteerIdentifier={makeNodeSteerIdentifier(node)}
                                  sourceId={makeNodeSourceId(node)}
                                  node={node}
                                  label={getOverrideClerpForNode(node) || ''}
                                  selectedGraph={selectedGraph}
                                  steeredPositions={steeredPositions}
                                  setSteeredPositions={setSteeredPositions}
                                  isSteered={isSteered}
                                  findSteeredPositionByPosition={findSteeredPositionByPosition}
                                  removeSteeredPosition={removeSteeredPosition}
                                  setSteeredPositionDeltaByPosition={setSteeredPositionDeltaByPosition}
                                  getDeltaToAddForMultiplier={getDeltaToAddForMultiplier}
                                  getTopActivationValue={getTopActivationValue}
                                  allTokensHaveSameDelta={allTokensHaveSameDelta}
                                  findSteeredPositionSteerGeneratedTokens={findSteeredPositionSteerGeneratedTokens}
                                  setSteeredPositionDeltaSteerGeneratedTokens={
                                    setSteeredPositionDeltaSteerGeneratedTokens
                                  }
                                  setAllTokensDelta={setAllTokensDelta}
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
              <div className="flex flex-1 flex-col gap-y-4 pb-0.5 pr-0.5">
                <Card className="flex w-full flex-col bg-white">
                  <CardHeader className="sticky top-0 flex w-full flex-row items-center justify-between rounded-t-xl bg-white pb-3 pt-5">
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
                        <div className="flex w-[70px] flex-row items-center gap-x-1 text-right text-[10px] font-medium uppercase leading-tight text-slate-400">
                          <div className="flex flex-col">
                            Freeze
                            <br />
                            Attention
                          </div>
                          <CustomTooltip
                            trigger={
                              <QuestionMarkIcon className="h-4 w-4 rounded-full bg-slate-200 p-1 text-slate-600" />
                            }
                          >
                            <div className="text-xs text-slate-700">
                              Freezing attention forces the attention pattern to be what it was in the original case,
                              with no steering.
                            </div>
                          </CustomTooltip>
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
                          if (steeredPositions.length === 0) {
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
                            features: steeredPositions,
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
                  <CardHeader className="sticky top-0 flex w-full flex-row items-center justify-between rounded-t-xl bg-white pb-3 pt-5">
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
