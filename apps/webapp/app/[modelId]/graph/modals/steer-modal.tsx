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
import { SteerLogitFeature, SteerLogitsRequest, SteerResponse } from '@/lib/utils/graph';
import { getLayerNumFromSource } from '@/lib/utils/source';
import {
  STEER_FREEZE_ATTENTION,
  STEER_FREQUENCY_PENALTY_GRAPH,
  STEER_FREQUENCY_PENALTY_MAX,
  STEER_FREQUENCY_PENALTY_MIN,
  STEER_N_COMPLETION_TOKENS_GRAPH,
  STEER_N_COMPLETION_TOKENS_GRAPH_MAX,
  STEER_SEED,
  STEER_TEMPERATURE_GRAPH,
  STEER_TEMPERATURE_MAX,
} from '@/lib/utils/steer';
import { ExplanationWithPartialRelations, NeuronWithPartialRelations } from '@/prisma/generated/zod';
import { QuestionMarkIcon, ResetIcon } from '@radix-ui/react-icons';
import { Joystick, Trash2 } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import NodeToSteer, { SteeredPositionIdentifier } from './steer-modal/node-to-steer';
import TokenTooltip from './steer-modal/token-tooltip';

const STEER_STRENGTH_ADDED_MULTIPLIER_CUSTOM_GRAPH = 1;

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
      const feature = findSteeredPositionByPosition(identifier, position);
      // feature not currently steered, add the steer at the specified position and delta, and also make it steer generated tokens
      if (!feature) {
        setSteeredPositions([
          ...steeredPositions,
          {
            layer: identifier.layer,
            index: identifier.index,
            delta: newDelta,
            steer_position: position,
            ablate,
            steer_generated_tokens: false,
            token_active_position: identifier.tokenActivePosition,
          },
        ]);
        return;
      }
      // feature is currently steered at this position, update the delta
      setSteeredPositions(
        steeredPositions.map((f) =>
          f.layer === identifier.layer &&
          f.index === identifier.index &&
          f.steer_position === position &&
          f.token_active_position === identifier.tokenActivePosition
            ? { ...f, delta: newDelta, ablate }
            : f,
        ),
      );
    },
    [findSteeredPositionByPosition, steeredPositions, setSteeredPositions],
  );

  const removeSteeredPosition = useCallback(
    (identifier: SteeredPositionIdentifier) => {
      setSteeredPositions(
        steeredPositions.filter(
          (f) =>
            !(
              f.layer === identifier.layer &&
              f.index === identifier.index &&
              f.token_active_position === identifier.tokenActivePosition
            ),
        ),
      );
    },
    [steeredPositions, setSteeredPositions],
  );

  const resetSteerSettings = () => {
    setSteerResult(undefined);
    setIsSteering(false);
    setSteeredPositions([]);
    setCustomSteerNodes([]);
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
    setFreezeAttention(STEER_FREEZE_ATTENTION);
  };

  useEffect(() => {
    // reset everything when selected graph changes
    resetSteerSettings();
  }, [selectedGraph]);

  const makeNodeSteerIdentifier = (node: CLTGraphNode): SteeredPositionIdentifier => ({
    modelId,
    layer: getLayerFromAnthropicFeatureId(modelId, node.feature),
    index: getIndexFromAnthropicFeatureId(modelId, node.feature),
    tokenActivePosition: node.ctx_idx,
  });

  const makeNodeSourceId = (node: CLTGraphNode): string =>
    `${getLayerFromAnthropicFeatureId(modelId, node.feature)}-${MODEL_TO_SOURCESET_ID[selectedGraph?.metadata.scan as keyof typeof MODEL_TO_SOURCESET_ID]}`;

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
                          setCustomSteerNodes([...customSteerNodes, node]);
                          setSteeredPositions([
                            ...steeredPositions,
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
                        it was active, which should cause the steered output to make the feature less prominent. You can
                        also drag sliders to steer the feature at specific positions. Middle will ablate that feature.
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
                            setSteeredPositions([]);
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
                                    setCustomSteerNodes(
                                      customSteerNodes.filter((node) => node.nodeId !== customNode.nodeId),
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
                      visState.supernodes.map((supernode) => {
                        if (supernode.length === 0) {
                          return null;
                        }
                        return (
                          <div key={supernode.join('-')} className="mb-2 rounded-md bg-slate-50 px-4 py-3 pb-3.5">
                            <div className="mb-2.5 flex w-full flex-row items-center justify-between gap-x-1.5">
                              <div className="flex flex-row items-end gap-x-1.5 text-[13px]">
                                <div>{supernode[0]}</div>
                                <span className="text-[8px] text-slate-400">SUPERNODE</span>
                              </div>
                            </div>
                            <div className="flex flex-col gap-y-1.5">
                              {supernode.slice(1).map((id) => {
                                const node = selectedGraph?.nodes.find((n) => n.nodeId === id);
                                if (!node || !nodeTypeHasFeatureDetail(node)) {
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
                              const node = selectedGraph?.nodes.find((n) => n.nodeId === id);
                              if (!node || !nodeTypeHasFeatureDetail(node)) {
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
                  <CardHeader className="sticky top-0 flex w-full flex-row items-center justify-between rounded-t-xl bg-white pb-3 pt-6">
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
