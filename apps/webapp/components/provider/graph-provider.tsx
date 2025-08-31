'use client';

import {
  ANT_MODEL_ID_TO_NEURONPEDIA_MODEL_ID,
  AnthropicFeatureDetail,
  CLTGraph,
  CLTGraphNode,
  CltVisState,
  FilterGraphType,
  MODEL_DIGITS_IN_FEATURE_ID,
  MODEL_HAS_S3_DASHBOARDS,
  MODEL_TO_SOURCESET_ID,
  MODEL_WITH_NP_DASHBOARDS_NOT_YET_CANTOR,
  ModelToGraphMetadatasMap,
  cltModelToNumLayers,
  convertAnthropicFeatureToNeuronpediaSourceSet,
  formatCLTGraphData,
  getIndexFromCantorValue,
  getIndexFromFeatureAndGraph,
  getLayerFromCantorValue,
  isHideLayer,
  modelIdToModelDisplayName,
  nodeTypeHasFeatureDetail,
  parseGraphClerps,
  parseGraphSupernodes,
} from '@/app/[modelId]/graph/utils';
import {
  ActivationWithPartialRelations,
  GraphMetadataSubgraphWithPartialRelations,
  GraphMetadataWithPartialRelations,
  NeuronWithPartialRelations,
} from '@/prisma/generated/zod';
import { GraphMetadata } from '@prisma/client';
import debounce from 'lodash/debounce';
import { useSession } from 'next-auth/react';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import {
  Dispatch,
  ReactNode,
  SetStateAction,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { useGlobalContext } from './global-provider';
import { useGraphStateContext } from './graph-state-provider';

const ANTHROPIC_FEATURE_DETAIL_DOWNLOAD_BATCH_SIZE = 32;
const NEURONPEDIA_FEATURE_DETAIL_DOWNLOAD_BATCH_SIZE = 2048;
export const GRAPH_PREFETCH_ACTIVATIONS_COUNT = 3;
const DEFAULT_DENSITY_THRESHOLD = 0.99;
export const PREFERRED_EXPLANATION_TYPES_NAMES = ['np_max-act-logits', 'np_max-act'];

// Define the main graph context type (without modal state)
type GraphContextType = {
  modelIdToMetadataMap: ModelToGraphMetadatasMap;
  selectedModelId: string;
  selectedMetadataGraph: GraphMetadataWithPartialRelations | null;
  selectedGraph: CLTGraph | null;
  setSelectedModelId: (modelId: string) => void;
  setSelectedMetadataGraph: (graph: GraphMetadataWithPartialRelations | null) => void;
  setDefaultMetadataGraph: (modelOverride?: string) => void;
  setModelIdToMetadataMap: (metadata: ModelToGraphMetadatasMap) => void;
  getGraph: (graphSlug: string) => Promise<CLTGraph>;

  // isLoadingGraphData
  isLoadingGraphData: boolean;
  setIsLoadingGraphData: (isLoading: boolean) => void;

  // visState
  visState: CltVisState;
  setVisState: (newState: Partial<CltVisState>) => void;
  updateVisStateField: <K extends keyof CltVisState>(key: K, value: CltVisState[K]) => void;
  togglePin: (nodeId: string) => string[];

  // logitDiff
  logitDiff: string | null;
  setLogitDiff: (logitDiff: string | null) => void;

  // resetSelectedGraphToDefaultVisState
  resetSelectedGraphToDefaultVisState: () => void;

  // isEditingLabel
  isEditingLabel: boolean;
  setIsEditingLabel: (isEditingLabel: boolean) => void;

  // getOverrideClerpForNode
  getOverrideClerpForNode: (node: CLTGraphNode) => string | undefined;

  getNodeSupernodeAndOverrideLabel: (node: CLTGraphNode) => string;

  // makeTooltipText
  makeTooltipText: (node: CLTGraphNode) => string;

  // getOriginalClerpForNode
  getOriginalClerpForNode: (node: CLTGraphNode) => string | undefined;

  // Graph filtering setting
  filterGraphsSetting: FilterGraphType[];
  setFilterGraphsSetting: (setting: FilterGraphType[]) => void;
  shouldShowGraphToCurrentUser: (graph: GraphMetadata) => boolean;

  // Loading graph label
  loadingGraphLabel: string;
  setLoadingGraphLabel: (label: string) => void;

  // setFullNPFeatureDetail
  setFullNPFeatureDetail: (setNode: Dispatch<SetStateAction<CLTGraphNode | null>>, node: CLTGraphNode) => void;

  // resetSelectedGraphToBlankVisState
  resetSelectedGraphToBlankVisState: () => void;

  loadSubgraph: (subgraph: GraphMetadataSubgraphWithPartialRelations) => void;
};

// Create the context with a default value
const GraphContext = createContext<GraphContextType | undefined>(undefined);

export function getGraphUrl(graphSlug: string, baseUrl: string): string {
  const url = `${baseUrl}/graph_data/${graphSlug}.json`;
  return url;
}

async function fetchInBatches<T>(
  items: any[],
  fetchFn: (item: any, signal?: AbortSignal) => Promise<T>,
  batchSize: number,
  abortSignal?: AbortSignal,
): Promise<T[]> {
  const results: T[] = [];
  for (let i = 0; i < items.length; i += batchSize) {
    if (abortSignal?.aborted) {
      throw new Error('Request cancelled in fetchInBatches');
    }
    const batchItems = items.slice(i, i + batchSize);
    const batchPromises = batchItems.map((item) => fetchFn(item, abortSignal));
    const batchResults = await Promise.all(batchPromises);
    results.push(...batchResults);
  }
  return results;
}

// Provider component
export function GraphProvider({
  children,
  initialModelIdToMetadataGraphsMap = {},
  initialModel,
  initialMetadataGraph,
  initialPinnedIds,
  initialSupernodes,
  initialClerps,
  initialPruningThreshold,
  initialDensityThreshold,
}: {
  children: ReactNode;
  initialModelIdToMetadataGraphsMap?: ModelToGraphMetadatasMap;
  initialModel?: string;
  initialMetadataGraph?: GraphMetadata;
  initialPinnedIds?: string;
  initialSupernodes?: string[][];
  initialClerps?: string[][];
  initialPruningThreshold?: number;
  initialDensityThreshold?: number;
}) {
  const router = useRouter();
  const session = useSession();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const { globalModels } = useGlobalContext();
  const { clickedIdRef } = useGraphStateContext();

  const [modelIdToMetadataMap, setModelIdToMetadataMap] = useState<ModelToGraphMetadatasMap>(
    initialModelIdToMetadataGraphsMap,
  );
  const [selectedModelId, setSelectedModelId] = useState<string>(
    initialModel ||
      (Object.keys(initialModelIdToMetadataGraphsMap).length > 0
        ? Object.keys(initialModelIdToMetadataGraphsMap)[0]
        : ''),
  );
  const [selectedMetadataGraph, setSelectedMetadataGraph] = useState<GraphMetadata | null>(
    initialMetadataGraph ||
      (selectedModelId && initialModelIdToMetadataGraphsMap[selectedModelId]?.length > 0
        ? initialModelIdToMetadataGraphsMap[selectedModelId][0]
        : null),
  );
  const [selectedGraph, setSelectedGraph] = useState<CLTGraph | null>(null);
  const [isLoadingGraphData, setIsLoadingGraphData] = useState<boolean>(true);
  const [filterGraphsSetting, setFilterGraphsSetting] = useState<FilterGraphType[]>([
    FilterGraphType.Featured,
    FilterGraphType.Community,
    FilterGraphType.Mine,
  ]);
  const [loadingGraphLabel, setLoadingGraphLabel] = useState<string>('');

  const hasAppliedInitialOverrides = useRef(false);
  const [isEditingLabel, setIsEditingLabel] = useState<boolean>(false);

  // Ref for the current AbortController
  const currentAbortController = useRef<AbortController | null>(null);

  const [visState, setVisStateInternal] = useState<CltVisState>({
    pinnedIds: initialPinnedIds ? initialPinnedIds.split(',') : [],
    hiddenIds: [],
    hoveredNodeId: null,
    hoveredCtxIdx: null,
    clickedCtxIdx: null,
    linkType: 'both',
    isShowAllLinks: '',
    isSyncEnabled: '',
    subgraph: null,
    isEditMode: 1,
    isHideLayer: false,
    sg_pos: '',
    isModal: true,
    isGridsnap: false,
    supernodes: initialSupernodes || [],
    clerps: initialClerps || [],

    densityThreshold: 1,
  });

  const getOriginalClerpForNode = (node: CLTGraphNode) => {
    if (node.featureDetailNP) {
      // if any of the explanations.typeName === PREFERRED_EXPLANATION_TYPE_NAME, then use that one
      const explanation = node.featureDetailNP.explanations?.find((e) =>
        PREFERRED_EXPLANATION_TYPES_NAMES.includes(e.typeName || ''),
      );
      if (explanation) {
        return explanation.description;
      }
      // otherwise just return the first explanation
      return node.featureDetailNP.explanations?.[0]?.description || '';
    }
    return node.ppClerp;
  };

  const getOverrideClerpForNode = (node: CLTGraphNode) => {
    const defaultClerp = getOriginalClerpForNode(node);
    if (visState.clerps && visState.clerps.length > 0) {
      const overrideClerp = visState.clerps.find((e) => e[0] === node.featureId);
      return overrideClerp ? overrideClerp[1] : defaultClerp;
    }
    return defaultClerp;
  };

  const getNodeSupernodeLabel = (node: CLTGraphNode) => {
    // look in visState.subgraph.supernodes array to check which array item includes this nodeid
    const supernode = visState.subgraph?.supernodes.find(
      (sn: string[]) => node.nodeId !== undefined && sn.includes(node.nodeId),
    );
    if (supernode) {
      return `[${supernode.length > 0 ? supernode[0] : ''}] `;
    }
    return '';
  };

  const getNodeSupernodeAndOverrideLabel = (node: CLTGraphNode) => {
    const supernodeLabel = getNodeSupernodeLabel(node);
    const overrideClerp = getOverrideClerpForNode(node);
    // some bug where the supernode label occurs twice
    if (supernodeLabel.length > 0 && overrideClerp !== undefined && overrideClerp?.startsWith(supernodeLabel)) {
      return overrideClerp;
    }
    if (supernodeLabel === overrideClerp) {
      return supernodeLabel;
    }
    return supernodeLabel + overrideClerp;
  };

  const makeTooltipText = (node: CLTGraphNode) => {
    const label = getNodeSupernodeAndOverrideLabel(node);
    return `${label.length === 0 ? 'Unlabeled' : getOverrideClerpForNode(node)} <br/> ${node.layer === 'E' ? 'Emb' : node.layer === 'Lgt' ? 'Logit' : `${node.isSuperNode ? '' : `Layer ${node.layer}`}`}`;
  };

  const getFilterGraphTypeForCurrentUser = (graph: GraphMetadata) => {
    if (session.data?.user?.id === graph.userId) {
      return FilterGraphType.Mine;
    }
    if (graph.isFeatured) {
      return FilterGraphType.Featured;
    }
    return FilterGraphType.Community;
  };

  const shouldShowGraphToCurrentUser = (graph: GraphMetadata) => {
    const graphType = getFilterGraphTypeForCurrentUser(graph);
    return filterGraphsSetting.includes(graphType);
  };

  const setDefaultMetadataGraph = (modelOverride?: string) => {
    console.log('setting default metadata graph');
    // based on the filter, find the first graph that matches the filter
    const filteredGraphs = modelIdToMetadataMap[modelOverride || selectedModelId].filter((graph) =>
      filterGraphsSetting.includes(getFilterGraphTypeForCurrentUser(graph)),
    );
    if (filteredGraphs.length > 0) {
      console.log('filteredGraphs:', filteredGraphs[0].slug);
      setSelectedMetadataGraph(filteredGraphs[0]);
    } else {
      setSelectedMetadataGraph(null);
    }
  };

  useEffect(() => {
    if (hasAppliedInitialOverrides.current) {
      setDefaultMetadataGraph();
    }
  }, [filterGraphsSetting]);

  const updateUrlParams = useCallback(
    (keysToValues: Record<string, string | null>) => {
      const params = new URLSearchParams(searchParams.toString());
      const originalParamsString = params.toString();

      Object.entries(keysToValues).forEach(([key, value]) => {
        if (value) {
          params.set(key, value);
        } else {
          params.delete(key);
        }
      });

      const newParamsString = params.toString();
      if (originalParamsString !== newParamsString) {
        router.replace(`${pathname}?${params.toString()}`, { scroll: false });
      }
    },
    [router, pathname, searchParams],
  );

  // update qParams when visState changes
  useEffect(() => {
    if (selectedGraph) {
      // Prioritize clickedId from graph state provider, fallback to visState
      const newParams = {
        slug: selectedMetadataGraph?.slug || null,
        pinnedIds: visState.pinnedIds.join(','),
        // clickedId: clickedIdRef.current || null,
        supernodes:
          visState.subgraph && visState.subgraph.supernodes.length > 0
            ? JSON.stringify(visState.subgraph.supernodes)
            : null,
        clerps: visState.clerps && visState.clerps.length > 0 ? JSON.stringify(visState.clerps) : null,
        pruningThreshold: visState.pruningThreshold?.toString() || null,
        densityThreshold: visState.densityThreshold?.toString() || null,
      };
      updateUrlParams(newParams);
    }
  }, [visState, clickedIdRef.current]);

  // TODO: this does not seem to be used - but in the original it had it in the url params
  const [logitDiff, setLogitDiff] = useState<string | null>(null);

  const blankVisState = {
    pinnedIds: [],
    hiddenIds: [],
    hoveredNodeId: null,
    hoveredCtxIdx: null,
    clickedId: null,
    clickedCtxIdx: null,
    linkType: 'both',
    isShowAllLinks: '',
    isSyncEnabled: '',
    subgraph: null,
    isEditMode: 1,
    isHideLayer: false,
    sg_pos: '',
    isModal: true,
    isGridsnap: false,
    supernodes: [],
    clerps: [],
  };

  function getGraphDefaultVisState(graph: CLTGraph) {
    let visStateToReturn: CltVisState = blankVisState;
    visStateToReturn.isHideLayer = isHideLayer(graph.metadata.scan);

    // if we have qParams (default queryparams/visstate), parse them as visState and set it
    if (graph.qParams) {
      visStateToReturn = {
        ...visStateToReturn,
        ...graph.qParams,
      };
    }
    visStateToReturn.pruningThreshold = 0.6;

    visStateToReturn.densityThreshold = 1;

    return visStateToReturn;
  }

  function resetSelectedGraphToBlankVisState() {
    // default vis state is either saved qParams or blank
    setVisStateInternal(blankVisState);
  }

  function resetSelectedGraphToDefaultVisState() {
    // default vis state is either saved qParams or blank
    if (selectedGraph) {
      setVisStateInternal(getGraphDefaultVisState(selectedGraph));
    }
  }

  useEffect(() => {
    if (selectedGraph) {
      const visStateToSet = getGraphDefaultVisState(selectedGraph);

      if (!hasAppliedInitialOverrides.current) {
        console.log('applying initial overrides');
        // apply overrides with each state value from initial values
        // only do this one time (first load from url)
        // we don't want to reapply these values when we switch graphs

        // override pinnedIds
        if (initialPinnedIds) {
          visStateToSet.pinnedIds = initialPinnedIds.split(',');
        }

        // override supernodes
        if (initialSupernodes) {
          if (visStateToSet.subgraph) {
            visStateToSet.subgraph.supernodes = initialSupernodes;
          } else {
            visStateToSet.subgraph = {
              supernodes: initialSupernodes,
              sticky: true,
              dagrefy: true,
              activeGrouping: {
                isActive: false,
                selectedNodeIds: new Set(),
              },
            };
          }
        }

        // override clerps
        if (initialClerps) {
          visStateToSet.clerps = initialClerps;
        }

        if (initialPruningThreshold) {
          visStateToSet.pruningThreshold = initialPruningThreshold;
        }

        if (initialDensityThreshold !== undefined) {
          visStateToSet.densityThreshold = initialDensityThreshold;
        } else {
          visStateToSet.densityThreshold = DEFAULT_DENSITY_THRESHOLD;
        }

        hasAppliedInitialOverrides.current = true;
      } else {
        console.log('not applying initial overrides');
      }

      setVisStateInternal(visStateToSet);
    }
  }, [selectedGraph]);

  // Function to update the entire visState
  const setVisState = (newState: Partial<CltVisState>) => {
    setVisStateInternal((prevState) => ({ ...prevState, ...newState }));
  };

  // Pin/Unpin the nodeId and return the new PinnedIds array
  // If it's in a supernode, we need to remove it from the supernode
  const togglePin = (nodeId: string): string[] => {
    const currentPinnedIds = visState.pinnedIds || [];
    const newPinnedIds = currentPinnedIds.includes(nodeId)
      ? currentPinnedIds.filter((id) => id !== nodeId)
      : [...currentPinnedIds, nodeId];

    setVisStateInternal((prevState) => ({
      ...prevState,
      pinnedIds: newPinnedIds,
    }));

    // if we are REMOVING the pin, and the nodeId is in a supernode, we need to remove it from the supernode
    if (!newPinnedIds.includes(nodeId) && visState.subgraph?.supernodes.some((sn) => sn.includes(nodeId))) {
      setVisStateInternal((prevState) => ({
        ...prevState,
        ...(prevState.subgraph && {
          subgraph: {
            ...prevState.subgraph,
            supernodes: prevState.subgraph.supernodes
              .map((sn) => {
                // If this supernode contains the nodeId, remove it
                if (sn.includes(nodeId)) {
                  return sn.filter((id) => id !== nodeId);
                }
                return sn;
              })
              .filter((sn) => sn.length > 2), // Remove supernodes with 2 or fewer items (first is label, second is a node. if there's only 2, then it's a single node)
            activeGrouping: {
              isActive: false,
              selectedNodeIds: new Set(),
            },
          },
        }),
      }));
    }

    return newPinnedIds;
  };

  // Function to update a single field of visState
  const updateVisStateField = useCallback(<K extends keyof CltVisState>(key: K, value: CltVisState[K]) => {
    setVisStateInternal((prevState) => ({ ...prevState, [key]: value }));
  }, []);

  const loadSubgraph = (subgraph: GraphMetadataSubgraphWithPartialRelations) => {
    setVisStateInternal(() => ({
      ...blankVisState,
      clerps: parseGraphClerps(JSON.stringify(subgraph.clerps)),
      pinnedIds: subgraph.pinnedIds,
      subgraph: {
        supernodes: parseGraphSupernodes(JSON.stringify(subgraph.supernodes)),
        sticky: true,
        dagrefy: true,
        activeGrouping: {
          isActive: false,
          selectedNodeIds: new Set(),
        },
      },
      pruningThreshold: subgraph.pruningThreshold || undefined,
      densityThreshold: subgraph.densityThreshold || undefined,
    }));
  };

  async function fetchAnthropicFeatureDetail(
    modelId: string,
    feature: number,
    baseUrl: string,
    abortSignal?: AbortSignal,
  ): Promise<AnthropicFeatureDetail | null> {
    try {
      const response = await fetch(`${baseUrl}/features/${modelId}/${feature}.json`, { signal: abortSignal });
      if (abortSignal?.aborted) {
        throw new Error('Request cancelled in fetchAnthropicFeatureDetail');
      }
      if (!response.ok) {
        console.error(`Failed to fetch feature detail for ${modelId}/${feature}`);
        return null;
      }
      const data = await response.json();
      return data as AnthropicFeatureDetail;
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        console.log('Fetch aborted in fetchAnthropicFeatureDetail');
        throw error;
      }
      console.error(`Error in fetchAnthropicFeatureDetail for ${modelId}/${feature}:`, error);
      return null;
    }
  }

  async function fetchFeatureDetailFromBaseURL(
    baseUrl: string,
    featureFilename: string,
    abortSignal?: AbortSignal,
  ): Promise<AnthropicFeatureDetail | null> {
    try {
      const response = await fetch(`${baseUrl}/${featureFilename}.json`, { signal: abortSignal });
      if (abortSignal?.aborted) {
        throw new Error('Request cancelled in fetchFeatureDetailFromBaseURL');
      }
      if (!response.ok) {
        console.error(`Failed to fetch feature detail for ${featureFilename}`);
        return null;
      }
      const data = await response.json();
      return data as AnthropicFeatureDetail;
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        console.log('Fetch aborted in fetchFeatureDetailFromBaseURL');
        throw error;
      }
      console.error(`Error in fetchFeatureDetailFromBaseURL for ${featureFilename}:`, error);
      return null;
    }
  }

  const getAnthropicBaseUrlFromGraphUrl = (url: string) => url.split('/graph_data/')[0];

  // Function to fetch graph data
  async function getGraph(graphSlug: string, abortSignal?: AbortSignal): Promise<CLTGraph> {
    const graph = modelIdToMetadataMap[selectedModelId]?.find((g) => g.slug === graphSlug);
    if (!graph) {
      throw new Error(`Graph not found for ${graphSlug}`);
    }

    // First, get the file size using a HEAD request
    const headResponse = await fetch(graph.url, { method: 'HEAD', signal: abortSignal });
    const contentLength = headResponse.headers.get('content-length');
    const fileSizeInMB = contentLength ? (parseInt(contentLength, 10) / (1024 * 1024)).toFixed(2) : 'unknown';

    setLoadingGraphLabel(`Loading Graph (${fileSizeInMB} MB)... `);

    if (abortSignal?.aborted) {
      throw new Error('Request cancelled before main graph fetch');
    }

    const response = await fetch(graph.url, { signal: abortSignal });

    if (!response.ok) {
      throw new Error(`Failed to fetch graph data for ${graphSlug}`);
    }

    // check if we have this model in cltModelToNumLayers
    // if selectedModelId is not in cltModelToNumLayers, then we need to get the num_layers from the database
    let numLayers = 0;
    if (selectedModelId in cltModelToNumLayers) {
      numLayers = cltModelToNumLayers[selectedModelId as keyof typeof cltModelToNumLayers];
    } else {
      const model = globalModels[selectedModelId];
      if (model) {
        numLayers = model.layers;
      }
    }

    let displayName = '';
    if (selectedModelId in modelIdToModelDisplayName) {
      displayName = modelIdToModelDisplayName.get(selectedModelId) || '';
    } else {
      displayName = globalModels[selectedModelId]?.displayName || '';
    }

    const dataJson = await response.json();
    if (abortSignal?.aborted) {
      throw new Error('Request cancelled after main graph JSON parsing');
    }
    const data = dataJson as CLTGraph;
    const formattedData = formatCLTGraphData(data, logitDiff);

    formattedData.metadata.neuronpedia_internal_model = {
      id: selectedModelId,
      displayName,
      layers: numLayers,
    };

    const isCantor = data.metadata.schema_version === 1 || data.metadata.feature_details?.neuronpedia_source_set;
    // if it specifies source_set, then it's cantor
    if (isCantor) {
      const model = data.metadata.scan;

      if (data.metadata.feature_details?.neuronpedia_source_set || model in MODEL_TO_SOURCESET_ID) {
        const sourceSet =
          data.metadata.feature_details?.neuronpedia_source_set ||
          MODEL_TO_SOURCESET_ID[model as keyof typeof MODEL_TO_SOURCESET_ID];

        // make an array of features to call /api/features
        // for neuronpedia fetches we only get the first 10 and then load more on demand
        const features = formattedData.nodes
          .filter((d) => nodeTypeHasFeatureDetail(d))
          .map((d) => {
            let layerNum = -1;
            let index = -1;

            // convert from feature id to layer and index using cantor pairing
            layerNum = parseInt(d.layer, 10);
            index = getIndexFromCantorValue(d.feature);
            return {
              modelId: model,
              layer: `${layerNum}-${sourceSet}`,
              index,
              maxActsToReturn: GRAPH_PREFETCH_ACTIVATIONS_COUNT,
            };
          });

        // split the features into batches of NEURONPEDIA_FEATURE_DETAIL_DOWNLOAD_BATCH_SIZE
        const batches = [];
        for (let i = 0; i < features.length; i += NEURONPEDIA_FEATURE_DETAIL_DOWNLOAD_BATCH_SIZE) {
          batches.push(features.slice(i, i + NEURONPEDIA_FEATURE_DETAIL_DOWNLOAD_BATCH_SIZE));
        }

        // call /api/features in batches, sequentially
        const batchesOfDetails = [];
        setLoadingGraphLabel(`Loading ${features.length} Nodes... `);
        for (const batch of batches) {
          const resp = await fetch('/api/features', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(batch),
            signal: abortSignal,
          });
          if (abortSignal?.aborted) {
            throw new Error('Request cancelled after /api/features batch fetch');
          }
          const da = (await resp.json()) as NeuronWithPartialRelations[];
          batchesOfDetails.push(da);
        }

        // put the details in the nodes
        const featureDetails = batchesOfDetails.flat(1);
        formattedData.nodes.forEach((d) => {
          // eslint-disable-next-line no-param-reassign
          const feature = featureDetails.find(
            (f) =>
              f &&
              'index' in f &&
              f.index === getIndexFromCantorValue(d.feature).toString() &&
              'layer' in f &&
              f.layer === `${d.layer}-${sourceSet}`,
          );
          if (feature) {
            // eslint-disable-next-line no-param-reassign
            d.featureDetailNP = feature as NeuronWithPartialRelations;
          }
        });
        // dont have sourceSet/NP yet, but is cantor
      } else {
        // currently only qwen3-4b mwhanna noskip transcoder is at this condition because it's:
        // 1) schema version 1
        // 2) is NOT in MODEL_TO_SOURCESET_ID (doesn't yet have neuronpedia dashboards)
        const mwhannaQwenNoSkipTranscoderFeaturePrefix = 'Qwen3-4b-relu-noskip-';
        const featureDetails = await fetchInBatches(
          formattedData.nodes,
          (d: CLTGraphNode, signal?: AbortSignal) => {
            if (nodeTypeHasFeatureDetail(d)) {
              return fetchFeatureDetailFromBaseURL(
                `https://d1fk9w8oratjix.cloudfront.net/features/Qwen3-4B/${mwhannaQwenNoSkipTranscoderFeaturePrefix}${getLayerFromCantorValue(d.feature)}`,
                getIndexFromCantorValue(d.feature).toString(),
                signal,
              );
            }
            return Promise.resolve(null);
          },
          ANTHROPIC_FEATURE_DETAIL_DOWNLOAD_BATCH_SIZE,
          abortSignal,
        );

        formattedData.nodes.forEach((d, i) => {
          // eslint-disable-next-line no-param-reassign
          d.featureDetail = featureDetails[i] as AnthropicFeatureDetail;
        });
      }
    }
    // TODO: remove this exception once fellows graph gets on cantor
    else if (MODEL_WITH_NP_DASHBOARDS_NOT_YET_CANTOR.has(selectedModelId)) {
      const model =
        ANT_MODEL_ID_TO_NEURONPEDIA_MODEL_ID[selectedModelId as keyof typeof ANT_MODEL_ID_TO_NEURONPEDIA_MODEL_ID];

      // make an array of features to call /api/features
      // for neuronpedia fetches we only get the first 10 and then load more on demand
      const features = formattedData.nodes
        .filter((d) => nodeTypeHasFeatureDetail(d))
        .map((d) => ({
          modelId: model,
          layer: convertAnthropicFeatureToNeuronpediaSourceSet(
            selectedModelId as keyof typeof MODEL_DIGITS_IN_FEATURE_ID,
            d,
          ),
          index: getIndexFromFeatureAndGraph(selectedModelId, d, formattedData),
          maxActsToReturn: GRAPH_PREFETCH_ACTIVATIONS_COUNT,
        }));

      console.log('number of features:', features.length);
      // split the features into batches of NEURONPEDIA_FEATURE_DETAIL_DOWNLOAD_BATCH_SIZE
      const batches = [];
      for (let i = 0; i < features.length; i += NEURONPEDIA_FEATURE_DETAIL_DOWNLOAD_BATCH_SIZE) {
        batches.push(features.slice(i, i + NEURONPEDIA_FEATURE_DETAIL_DOWNLOAD_BATCH_SIZE));
      }

      // call /api/features in batches, sequentially
      const batchesOfDetails = [];
      setLoadingGraphLabel(`Loading ${features.length} Nodes... `);
      for (const batch of batches) {
        const resp = await fetch('/api/features', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(batch),
          signal: abortSignal,
        });
        if (abortSignal?.aborted) {
          throw new Error(
            'Request cancelled after /api/features batch fetch (MODEL_WITH_NP_DASHBOARDS_NOT_YET_CANTOR)',
          );
        }
        const da = (await resp.json()) as NeuronWithPartialRelations[];
        batchesOfDetails.push(da);
      }

      // put the details in the nodes
      const featureDetails = batchesOfDetails.flat(1);
      formattedData.nodes
        .filter((d) => nodeTypeHasFeatureDetail(d)) // sometimes there are duplicate feature numbers from embed / logits / mlp recon error, we should always filter them out
        .forEach((d) => {
          // eslint-disable-next-line no-param-reassign
          const feature = featureDetails.find(
            (f) =>
              f &&
              'index' in f &&
              f.index === getIndexFromFeatureAndGraph(selectedModelId, d, formattedData).toString() &&
              'layer' in f &&
              f.layer ===
                convertAnthropicFeatureToNeuronpediaSourceSet(
                  selectedModelId as keyof typeof MODEL_DIGITS_IN_FEATURE_ID,
                  d,
                ),
          );
          if (feature) {
            // eslint-disable-next-line no-param-reassign
            d.featureDetailNP = feature as NeuronWithPartialRelations;
          }
        });
    } else if (selectedModelId === 'llama-3.2-1b') {
      // special case this
      console.log('llama-3.2-1b, special case to get features');
      const featureDetails = await fetchInBatches(
        formattedData.nodes,
        (d, signal?: AbortSignal) => {
          if (nodeTypeHasFeatureDetail(d)) {
            return fetchAnthropicFeatureDetail(
              'llama-3-131k-relu',
              d.feature,
              'https://d1fk9w8oratjix.cloudfront.net',
              signal,
            );
          }
          return Promise.resolve(null);
        },
        ANTHROPIC_FEATURE_DETAIL_DOWNLOAD_BATCH_SIZE,
        abortSignal,
      );

      formattedData.nodes.forEach((d, i) => {
        // eslint-disable-next-line no-param-reassign
        d.featureDetail = featureDetails[i] as AnthropicFeatureDetail;
      });
    } else if (selectedModelId === 'qwen3-4b') {
      // these are the mntss skip transcoders
      const featureDetails = await fetchInBatches(
        formattedData.nodes,
        (d: CLTGraphNode, signal?: AbortSignal) => {
          if (nodeTypeHasFeatureDetail(d)) {
            return fetchFeatureDetailFromBaseURL(
              `https://d1fk9w8oratjix.cloudfront.net/features/Qwen3-4B/qwen3-4b-${d.layer}`,
              d.feature.toString(),
              signal,
            );
          }
          return Promise.resolve(null);
        },
        ANTHROPIC_FEATURE_DETAIL_DOWNLOAD_BATCH_SIZE,
        abortSignal,
      );

      formattedData.nodes.forEach((d, i) => {
        // eslint-disable-next-line no-param-reassign
        d.featureDetail = featureDetails[i] as AnthropicFeatureDetail;
      });
    } else if (MODEL_HAS_S3_DASHBOARDS.has(selectedModelId)) {
      // otherwise get the feature from the bucket
      const featureDetails = await fetchInBatches(
        formattedData.nodes,
        (d, signal?: AbortSignal) => {
          if (nodeTypeHasFeatureDetail(d)) {
            return fetchAnthropicFeatureDetail(
              selectedModelId,
              d.feature,
              getAnthropicBaseUrlFromGraphUrl(graph.url),
              signal,
            );
          }
          return Promise.resolve(null);
        },
        ANTHROPIC_FEATURE_DETAIL_DOWNLOAD_BATCH_SIZE,
        abortSignal,
      );

      formattedData.nodes.forEach((d, i) => {
        // eslint-disable-next-line no-param-reassign
        d.featureDetail = featureDetails[i] as AnthropicFeatureDetail;
      });
    }
    // neither neuronpedia nor s3 - check for feature_json_base_url
    else if (data.metadata.feature_details?.feature_json_base_url) {
      console.log('feature_json_base_url:', data.metadata.feature_details.feature_json_base_url);
      const featureDetails = await fetchInBatches(
        formattedData.nodes,
        (d: CLTGraphNode, signal?: AbortSignal) => {
          if (nodeTypeHasFeatureDetail(d)) {
            if (!data.metadata.feature_details?.feature_json_base_url) {
              return Promise.resolve(null);
            }
            return fetchFeatureDetailFromBaseURL(
              data.metadata.feature_details.feature_json_base_url,
              d.feature.toString(),
              signal,
            );
          }
          return Promise.resolve(null);
        },
        ANTHROPIC_FEATURE_DETAIL_DOWNLOAD_BATCH_SIZE,
        abortSignal,
      );

      formattedData.nodes.forEach((d, i) => {
        // eslint-disable-next-line no-param-reassign
        d.featureDetail = featureDetails[i] as AnthropicFeatureDetail;
      });
    }
    // TODO: remove these special cases
    // if the model is gpt2-small, then it's goodfire
    else if (data.metadata.scan === 'gpt2-small') {
      console.log('gpt2-small, using goodfire to get features');
      // download https://clt-frontend.goodfire.pub/features/gpt2/163053.json
      // otherwise get the feature from the bucket
      const featureDetails = await fetchInBatches(
        formattedData.nodes,
        (d: CLTGraphNode, signal?: AbortSignal) => {
          if (nodeTypeHasFeatureDetail(d)) {
            return fetchFeatureDetailFromBaseURL(
              'https://clt-frontend.goodfire.pub/features/gpt2',
              d.feature.toString(),
              signal,
            );
          }
          return Promise.resolve(null);
        },
        ANTHROPIC_FEATURE_DETAIL_DOWNLOAD_BATCH_SIZE,
        abortSignal,
      );

      formattedData.nodes.forEach((d, i) => {
        // eslint-disable-next-line no-param-reassign
        d.featureDetail = featureDetails[i] as AnthropicFeatureDetail;
      });
    }
    // if the model is gelu-4l-x128k64-v0, then it's eleuther
    else if (data.metadata.scan === 'gelu-4l-x128k64-v0') {
      console.log('gelu-4l-x128k64-v0, using eleuther to get features');
      //
    }
    // otherwise we dont fill it in bc we don't know how

    setIsLoadingGraphData(false);
    return formattedData;
  }

  // Fetch graph data when selected metadata graph changes
  useEffect(() => {
    // Cancel previous request if it exists
    if (currentAbortController.current) {
      currentAbortController.current.abort();
      console.log('Previous graph load cancelled');
    }

    if (selectedMetadataGraph) {
      console.log('selectedMetadataGraph:', selectedMetadataGraph.slug);
      setIsLoadingGraphData(true);

      // Create new abort controller for this request
      const abortController = new AbortController();
      currentAbortController.current = abortController;

      getGraph(selectedMetadataGraph.slug, abortController.signal)
        .then((g) => {
          // Only update state if this request wasn't cancelled
          if (!abortController.signal.aborted) {
            setSelectedGraph(g);
          } else {
            console.log(`Graph load for ${selectedMetadataGraph.slug} was aborted, not setting selected graph.`);
          }
        })
        .catch((error) => {
          // Only handle error if not cancelled
          if (!abortController.signal.aborted) {
            console.error('Error loading graph:', error);
            // Potentially reset loading state or show error message
            setIsLoadingGraphData(false); // Ensure loading state is reset on error too
            setSelectedGraph(null); // Clear graph if loading failed
          } else {
            console.log(`Error caught for aborted graph load (${selectedMetadataGraph.slug}):`, error.message);
          }
        });
    } else {
      updateUrlParams({
        slug: null,
      });
      setSelectedGraph(null);
      setIsLoadingGraphData(false); // No graph selected, not loading
    }

    // Cleanup function to abort on unmount or before next run
    return () => {
      if (currentAbortController.current) {
        currentAbortController.current.abort();
        console.log('Graph load cancelled on cleanup');
        currentAbortController.current = null; // Clear the ref
      }
    };
  }, [selectedMetadataGraph]); // Keep other dependencies if any, but selectedMetadataGraph is key

  const setFullNPFeatureDetail = useMemo(
    () =>
      debounce((setNode: Dispatch<SetStateAction<CLTGraphNode | null>>, node: CLTGraphNode) => {
        // load the rest of the activations on demand
        if (
          node?.featureDetailNP &&
          node?.featureDetailNP.activations &&
          node?.featureDetailNP.activations.length <= GRAPH_PREFETCH_ACTIVATIONS_COUNT
        ) {
          fetch(`/api/activation/get`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              modelId: node.featureDetailNP.modelId,
              source: node.featureDetailNP.layer,
              index: node.featureDetailNP.index,
            }),
          })
            .then((response) => response.json())
            .then((acts: ActivationWithPartialRelations[]) => {
              if (node?.featureDetailNP) {
                // console.log('setting full NP feature detail', acts.length);
                // Create a new node object with updated activations to trigger re-render
                setNode((prevNode: CLTGraphNode | null) => {
                  if (!prevNode || !prevNode.featureDetailNP) return prevNode;
                  // don't change node object if it's a different node now
                  if (prevNode.nodeId !== node?.nodeId) return prevNode;
                  return {
                    ...prevNode,
                    featureDetailNP: {
                      ...prevNode.featureDetailNP,
                      activations: acts,
                    },
                  };
                });
                // add it to the selectedGraph
                if (selectedGraph) {
                  const matchingNode = selectedGraph.nodes.find((n) => n.nodeId === node.nodeId);
                  if (matchingNode && matchingNode.featureDetailNP) {
                    matchingNode.featureDetailNP.activations = acts;
                  }
                }
              }
            })
            .catch((error) => {
              console.error(`error submitting getting rest of feature: ${error}`);
            });
        }
      }, 250),
    [selectedGraph],
  );

  // Provide the context value (without modal state)
  const contextValue = useMemo(
    () => ({
      modelIdToMetadataMap,
      selectedModelId,
      selectedMetadataGraph,
      setDefaultMetadataGraph,
      selectedGraph,
      setSelectedModelId,
      setSelectedMetadataGraph,
      setModelIdToMetadataMap,
      getGraph,
      visState,
      setVisState,
      updateVisStateField,
      togglePin,
      logitDiff,
      setLogitDiff,
      isLoadingGraphData,
      setIsLoadingGraphData,
      resetSelectedGraphToDefaultVisState,
      isEditingLabel,
      setIsEditingLabel,
      getOverrideClerpForNode,
      getNodeSupernodeAndOverrideLabel,
      getOriginalClerpForNode,
      makeTooltipText,
      filterGraphsSetting,
      setFilterGraphsSetting,
      shouldShowGraphToCurrentUser,
      loadingGraphLabel,
      setLoadingGraphLabel,
      setFullNPFeatureDetail,
      resetSelectedGraphToBlankVisState,
      loadSubgraph,
    }),
    [
      modelIdToMetadataMap,
      selectedModelId,
      selectedMetadataGraph,
      setDefaultMetadataGraph,
      selectedGraph,
      visState,
      logitDiff,
      updateVisStateField,
      togglePin,
      isLoadingGraphData,
      setIsLoadingGraphData,
      resetSelectedGraphToDefaultVisState,
      isEditingLabel,
      setIsEditingLabel,
      getOverrideClerpForNode,
      getNodeSupernodeAndOverrideLabel,
      getOriginalClerpForNode,
      makeTooltipText,
      filterGraphsSetting,
      shouldShowGraphToCurrentUser,
      loadingGraphLabel,
      setLoadingGraphLabel,
      setFullNPFeatureDetail,
      resetSelectedGraphToBlankVisState,
      loadSubgraph,
    ],
  );

  return <GraphContext.Provider value={contextValue}>{children}</GraphContext.Provider>;
}

// Custom hook to use the context
export function useGraphContext() {
  const context = useContext(GraphContext);
  if (context === undefined) {
    throw new Error('useGraphContext must be used within a GraphProvider');
  }
  return context;
}
