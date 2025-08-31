/* eslint-disable jsx-a11y/alt-text */
/* eslint-disable jsx-a11y/click-events-have-key-events */
/* eslint-disable jsx-a11y/no-static-element-interactions */

'use client';

import { useGraphModalContext } from '@/components/provider/graph-modal-provider';

import { Button } from '@/components/shadcn/button';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/shadcn/dialog';
import { ASSET_BASE_URL } from '@/lib/env';
import { MagicWandIcon } from '@radix-ui/react-icons';
import { BookOpen, ChevronRight, Circle, GithubIcon, NewspaperIcon, NotebookIcon } from 'lucide-react';
import { useSearchParams } from 'next/navigation';
import { useEffect, useState } from 'react';

const ALWAYS_SHOW_WELCOME_MODAL = false;

export default function WelcomeModal({ hasSlug }: { hasSlug: boolean }) {
  const {
    isWelcomeModalOpen,
    setIsWelcomeModalOpen,
    setIsGenerateGraphModalOpen,
    welcomeModalInitialStep,
    resetWelcomeModalStep,
  } = useGraphModalContext();
  const [currentStep, setCurrentStep] = useState(welcomeModalInitialStep || 0);
  const searchParams = useSearchParams();

  // Array of images for each step
  const stepImages = [
    `${ASSET_BASE_URL}/graph/explainer-new.jpg`, // Getting Started
    `${ASSET_BASE_URL}/graph/explainer-top.jpg`, // Choose or Generate a Graph
    `${ASSET_BASE_URL}/graph/tl.jpg`, // Link/Attribution Graph
    `${ASSET_BASE_URL}/graph/tr.jpg`, // Connections
    `${ASSET_BASE_URL}/graph/subgraph-demo.mp4`, // Subgraph
    `${ASSET_BASE_URL}/graph/br.jpg`, // Feature Details
  ];

  // Array of alt text for each step
  const stepImageAlts = [
    'Getting started with Circuit Tracer - overview of solved and unsolved graphs',
    'Graph generation interface showing how to create custom prompts',
    'Link/Attribution graph displaying model reasoning process with nodes and edges',
    'Connections panel showing input and output features for selected nodes',
    'Subgraph interface for pinning and grouping nodes into supernodes',
    'Feature details panel showing top activations and logits for selected features',
  ];

  function playVideoDemo(step: number) {
    const video = document.getElementById(`video-${step}`) as HTMLVideoElement;
    video.play();
  }

  const steps = [
    {
      number: 0,
      title: 'Overview',
      subtitle: 'Introduction',
      bgColor: 'bg-indigo-100',
      textColor: 'text-indigo-600',
      content: (
        <div className="flex flex-col gap-y-3 text-left text-[13px] leading-normal">
          How does a language model decide how to respond? Interactively trace its internal reasoning steps and generate
          your own graphs with custom prompts. To read the guide later, click below.
          <Button
            onClick={() => {
              setIsWelcomeModalOpen(false);
              setIsGenerateGraphModalOpen(true);
            }}
            className="text-medium h-10 w-full gap-x-2 bg-emerald-700 text-[12.5px] text-white shadow-none hover:bg-emerald-800"
          >
            <MagicWandIcon className="h-4 w-4" />
            Generate New Graph
          </Button>
          <p className="mt-2">
            After generating a graph, your objective is to create a <b>subgraph solution</b>. To do this, you pin and
            group nodes to create a subgraph that explains key internal reasoning steps. Examples:
          </p>
          <div className="pt-2 text-left text-[11px] font-medium uppercase leading-none text-slate-400">
            Solved Graphs
          </div>
          <ul className="-mt-2 ml-4 list-disc text-[12px]">
            <li className="">
              <div
                className="cursor-pointer text-sky-600 hover:underline"
                onClick={() => {
                  window.location.href =
                    '/gemma-2-2b/graph?slug=gemma-fact-dallas-austin&pruningThreshold=0.6&pinnedIds=27_22605_10%2C20_15589_10%2CE_26865_9%2C21_5943_10%2C23_12237_10%2C20_15589_9%2C16_25_9%2C14_2268_9%2C18_8959_10%2C4_13154_9%2C7_6861_9%2C19_1445_10%2CE_2329_7%2CE_6037_4%2C0_13727_7%2C6_4012_7%2C17_7178_10%2C15_4494_4%2C6_4662_4%2C4_7671_4%2C3_13984_4%2C1_1000_4%2C19_7477_9%2C18_6101_10%2C16_4298_10%2C7_691_10&supernodes=%5B%5B%22capital%22%2C%2215_4494_4%22%2C%226_4662_4%22%2C%224_7671_4%22%2C%223_13984_4%22%2C%221_1000_4%22%5D%2C%5B%22state%22%2C%226_4012_7%22%2C%220_13727_7%22%5D%2C%5B%22Texas%22%2C%2220_15589_9%22%2C%2219_7477_9%22%2C%2216_25_9%22%2C%224_13154_9%22%2C%2214_2268_9%22%2C%227_6861_9%22%5D%2C%5B%22preposition+followed+by+place+name%22%2C%2219_1445_10%22%2C%2218_6101_10%22%5D%2C%5B%22capital+cities+%2F+say+a+capital+city%22%2C%2221_5943_10%22%2C%2217_7178_10%22%2C%227_691_10%22%2C%2216_4298_10%22%5D%5D&densityThreshold=0.99&clerps=%5B%5B%2223_2312237_10%22%2C%22Cities+and+states+names+%28say+Austin%29%22%5D%2C%5B%2218_1808959_10%22%2C%22state+%2F+regional+government%22%5D%5D';
                  setIsWelcomeModalOpen(false);
                }}
              >
                capital of state containing Dallas → Austin{' '}
              </div>
            </li>
            <li>
              opposite of &quot;small&quot; is → &quot;big&quot; {` `}
              <span
                className="cursor-pointer text-sky-600 hover:underline"
                onClick={() => {
                  window.location.href =
                    // eslint-disable-next-line
                    '/gemma-2-2b/graph?slug=gemma-small-big-en&clerps=[]&pruningThreshold=0.65&pinnedIds=27_13210_8,E_10498_5,23_8683_8,21_10062_8,17_12530_5,18_9402_8,6_4362_5,15_5617_5,15_5756_5,19_5058_8,14_11360_5,E_13388_2,15_7209_2,4_95_2,3_6576_2,27_7773_8,7_10545_5&supernodes=[["Output+\"big\"+or+\"large\"","27_7773_8","27_13210_8"],["say+big+/+huge+/+large","21_10062_8","23_8683_8"],["opposite","4_95_2","15_7209_2","3_6576_2"],["small","14_11360_5","17_12530_5","15_5617_5"],["large+/+size","6_4362_5","7_10545_5","15_5756_5"]]&clickedId=6_4362_5';
                }}
              >
                Eng
              </span>
              ,{' '}
              <span
                className="cursor-pointer text-sky-600 hover:underline"
                onClick={() => {
                  window.location.href =
                    '/gemma-2-2b/graph?slug=gemma-small-big-fr&pruningThreshold=0.65&pinnedIds=27_21996_8,E_64986_5,24_16045_8,19_5058_8,21_10062_8,23_2592_8,20_1454_8,E_63265_2,23_8683_8,23_8488_8,20_11434_8,19_5802_8,E_1455_7,15_5617_5,18_9402_8,6_4362_5,14_11360_5,3_2908_5,2_5452_5,3_6627_5,6_16184_2,4_95_2,22_10566_8,21_1144_8,E_2025_1,E_581_3&supernodes=[["opposite","6_16184_2","4_95_2"],["say+big+/+large","23_8683_8","23_8488_8","21_10062_8"],["comparatives","19_5058_8","24_16045_8","20_11434_8"],["small","15_5617_5","14_11360_5","3_6627_5","3_2908_5","2_5452_5"],["size","18_9402_8","6_4362_5"],["French","21_1144_8","22_10566_8","20_1454_8","23_2592_8","19_5802_8"]]&clickedId=22_10566_8&densityThreshold=0.99';
                }}
              >
                Fra
              </span>
              ,{' '}
              <span
                className="cursor-pointer text-sky-600 hover:underline"
                onClick={() => {
                  window.location.href =
                    '/gemma-2-2b/graph?slug=gemma-small-big-zh&pruningThreshold=0.65&pinnedIds=27_235469_8,E_235585_2,23_8488_8,23_8683_8,21_10062_8,19_5058_8,22_11933_8,21_9377_8,18_9402_8,15_5617_2,14_11360_2,14_13476_2,2_2169_2,1_10169_2,8_1988_6,4_15846_6,4_7409_6,E_208659_4,E_237379_6,E_236711_5,24_2394_8,23_13630_8,21_13505_8,20_12983_8&supernodes=[["reverse","4_7409_6","8_1988_6","4_15846_6"],["small","15_5617_2","14_11360_2"],["say+big+/+large","23_8683_8","21_10062_8","23_8488_8"],["Chinese","24_2394_8","22_11933_8","20_12983_8","21_13505_8","23_13630_8"],["Chinese-related+English+text","1_10169_2","14_13476_2"],["size","18_9402_8","2_2169_2"],["comparatives","21_9377_8","19_5058_8"]]&clickedId=27_235469_8&densityThreshold=0.99';
                }}
              >
                Chn
              </span>
            </li>
            <li>
              <div
                className="cursor-pointer text-sky-600 hover:underline"
                onClick={() => {
                  window.location.href =
                    '/gemma-2-2b/graph?slug=gemma-girls-are&pinnedIds=27_708_6,25_9974_6,22_11517_6,E_8216_2,E_674_3,E_651_1,19_1880_6,15_13979_6,17_7377_6,18_703_6,16_3689_6,15_4906_6,15_233_6,E_17733_6,3_6616_6,6_11265_6,5_1034_6,4_2671_6,3_6243_4,3_9864_3,0_13503_3&clickedId=3_9864_3&supernodes=[["see/saw","15_233_6","6_11265_6","3_6616_6"],["ends+of+noun+phrases+(predict+a+verb)","19_1880_6","17_7377_6"],["verbs+ending+relative+clauses","4_2671_6","15_4906_6","15_13979_6","18_703_6"],["that","0_13503_3","3_9864_3"]]&pruningThreshold=0.7&densityThreshold=0.99&clerps=[["25_2509974_6","say+are"],["5_501034_6","transitive+verbs+with+objects+preceding+htem"]]';
                }}
              >
                The girls that the teacher sees → are
              </div>
            </li>
          </ul>
          <div className="pt-2 text-left text-[11px] font-medium uppercase leading-none text-slate-400">
            Unsolved Graphs
          </div>
          <ul className="-mt-2 ml-4 list-disc text-[12px]">
            <li>
              <div
                className="cursor-pointer text-sky-600 hover:underline"
                onClick={() => {
                  window.location.href =
                    '/gemma-2-2b/graph?slug=gemma-Mexico-Spanish&clickedId=undefined&pruningThreshold=0.7&densityThreshold=0.99';
                }}
              >
                spoken in country south of the US → Spanish
              </div>
            </li>
            <li>
              <div
                className="cursor-pointer text-sky-600 hover:underline"
                onClick={() => {
                  window.location.href = '/gemma-2-2b/graph?slug=gemma-bee-insect&clerps=[]';
                }}
              >
                A bee is a type of → insect
              </div>
            </li>
            <li>
              <div
                className="cursor-pointer text-sky-600 hover:underline"
                onClick={() => {
                  window.location.href = '/gemma-2-2b/graph?slug=gemma-cat-hat&clerps=[]';
                }}
              >
                cat, bat, hat → rat
              </div>
            </li>
          </ul>
          <p className="mt-2.5 text-center">
            Click <strong>Next</strong> below to continue.
          </p>
        </div>
      ),
    },
    {
      number: 1,
      title: 'Top Toolbar',
      subtitle: 'Generate Graph',
      bgColor: 'bg-blue-100',
      textColor: 'text-blue-600',
      content: (
        <div className="flex flex-col gap-y-3 text-left text-sm">
          <p className="">
            Each graph is generated by showing the model a prompt and examining its internals as it processes each part
            of the prompt.
          </p>
          <p>
            You can <strong>generate your own graph</strong> by clicking on &quot;New Graph&quot; and entering your own
            prompt.
          </p>
          <p>
            In this example, we&apos;ve chosen a graph where we give the text &quot;Fact: The capital of the state
            containing Dallas is&quot;, to see how it comes up with &quot;Austin&quot;. In general, you want your prompt
            to be &quot;missing&quot; a word at the end, because we want to analyze how the model comes up with that
            word.
          </p>
        </div>
      ),
    },
    {
      number: 2,
      title: 'Top Left',
      subtitle: 'Attribution Graph',
      bgColor: 'bg-green-100',
      textColor: 'text-green-600',
      content: (
        <div className="flex flex-col gap-y-3 text-left text-[13px]">
          <p className="">
            The link (or attribution) graph at the top left displays the model&apos;s reasoning process as a graph.
            Intermediate steps are represented as nodes, with edges indicating the effect of one node on another.
          </p>
          <p>
            Here, we can see the prompt we entered, broken up into tokens, on the x axis of the graph. And, we&apos;ve
            clicked on a node labeled &quot;Texas&quot; in layer 20, which highlights it with a pink border and shows
            edges connecting to other nodes.
          </p>
          <p>
            The prompt&apos;s <strong>input tokens</strong> are the bottom nodes ({`"<bos>"`}, &quot;Fact&quot;,
            &quot;:&quot;, &quot;The&quot;, etc).
          </p>
          <p>
            The model&apos;s most likely responses are the <strong>&quot;output&quot; nodes</strong> at the top right (
            {`"San"`}, &quot;Oklahoma, &quot;Dallas&quot;, etc).
          </p>
          <p>
            The nodes in between correspond to ● <strong>features</strong>, which reflect concepts the model represents
            in its internal activations.
          </p>
          <p>
            We also show <strong>◆ error nodes</strong>, which are &quot;missing pieces&quot; not captured by our
            algorithm.
          </p>
        </div>
      ),
    },
    {
      number: 3,
      title: 'Top Right',
      subtitle: 'Node Connections',
      bgColor: 'bg-purple-100',
      textColor: 'text-purple-600',
      content: (
        <div className="flex flex-col gap-y-3 text-left text-sm">
          <p className="">
            When you click on a ● feature node in the link/attribution graph, you&apos;ll see its label and its
            connected nodes are displayed at the top right panel.
          </p>
          <p>The connected nodes are sorted by weight and separated into input features and output features.</p>
          <p>
            Since we clicked on &quot;Texas&quot; in layer 20, we see the lists of its connected nodes like
            &quot;Dallas&quot;, &quot;state/regional government&quot;, and &quot;Texas legal documents&quot;.
          </p>
        </div>
      ),
    },
    {
      number: 4,
      title: 'Bottom Left',
      subtitle: 'Subgraph',
      bgColor: 'bg-yellow-100',
      textColor: 'text-yellow-600',
      content: (
        <div className="flex flex-col gap-y-3 text-left text-[13px]">
          <p className="">
            The subgraph is a scratchpad for pinning and grouping nodes to make sense of the link graph. It&apos;s where
            you create a &quot;solution&quot; to what the model is thinking to arrive at its output token.
          </p>
          <p>
            In the demo to the right, we pin the nodes for &quot;Texas&quot;, &quot;cities&quot;, and
            &quot;Dallas&quot;. Then, we group &quot;cities&quot; and &quot;Dallas&quot;, and named that group
            &quot;cities&quot;. Finally, we save the subgraph as &quot;my subgraph 2&quot;.
          </p>
          <Button
            variant="slateLight"
            onClick={() => {
              playVideoDemo(4);
            }}
            size="xs"
            className="rounded-full bg-sky-100 px-3 py-2 text-xs text-sky-700 hover:bg-sky-200 hover:text-sky-700"
          >
            Play Video Demo
          </Button>
          <div className="flex flex-col gap-y-3 text-sm">
            <div className="items-left flex w-full flex-col gap-y-0.5">
              <strong>Pin Node to Subgraph</strong>
              <div className="ml-0 text-xs">
                Click a <Circle className="mb-1 mr-0.5 inline h-3 w-3" />
                node in the link graph, then click <strong>Pin Node</strong>. Alternatively, Cmd+Click a node to pin it.
              </div>
            </div>
            <div className="items-left flex w-full flex-col gap-y-0.5">
              <strong>Grouping</strong>
              <div className="ml-0 text-left text-xs">
                Click <strong>Grouping Mode</strong>, select subgraph nodes, and click <strong>Save Group</strong>.
                Alternatively, hold &apos;g&apos;, click multiple nodes, and release to group them.
              </div>
            </div>
            <div className="items-left flex w-full flex-col gap-y-0.5">
              <strong>Label Group</strong>
              <div className="ml-0 text-xs">Click the label under a group to give it a custom label.</div>
            </div>
            <div className="items-left flex w-full flex-col gap-y-0.5">
              <strong>Save & Share</strong>
              <div className="ml-0 text-xs">Use the tools at the top left to load, save, and share your subgraph.</div>
            </div>
          </div>
        </div>
      ),
    },
    {
      number: 5,
      title: 'Bottom Right',
      subtitle: 'Feature Details',
      bgColor: 'bg-indigo-100',
      textColor: 'text-indigo-600',
      content: (
        <div className="flex flex-col gap-y-3 text-left text-sm">
          <p className="">
            When you hover over or click a node in any of the panels, its feature details will be displayed here.
          </p>
          <p>
            The <strong>top activations</strong> show the contexts in a dataset that most strongly activated a feature.
            Finding the pattern in these contexts helps you determine what a feature represents
          </p>
          <p>
            The <strong>logits</strong> tell you the output tokens that the feature most strongly pushes the model to
            say, via direct connections. For later-layer features, these are often the best way to understand what a
            feature does. For earlier layer features, they can be misleading.
          </p>
          <p>
            Here, since we&apos;ve clicked the &quot;Texas&quot; node in the link/attribution graph, we see its top
            activations, logits, feature density, and histogram.
          </p>
          <p>You can also rename labels in the feature details panel, by clicking &quot;Edit Label&quot;.</p>
        </div>
      ),
    },
  ];

  useEffect(() => {
    try {
      // Don't show modal on mobile screens (less than 640px width, which is sm breakpoint)
      const isMobile = window.innerWidth < 640;
      if (isMobile) {
        return;
      }

      // // Don't show modal on start if we had a slug on first load (user just wanted to go directly to a graph)
      if (hasSlug) {
        return;
      }

      // Don't show modal when in embed mode
      const isEmbed = searchParams.get('embed') === 'true';
      if (isEmbed) {
        return;
      }

      const hasVisited = localStorage.getItem('circuit-tracer-visited');
      if (!hasVisited || ALWAYS_SHOW_WELCOME_MODAL) {
        setIsWelcomeModalOpen(true);
      }
    } catch (error) {
      console.error('Error checking localStorage:', error);
    }
  }, [searchParams]);

  // Update currentStep when welcomeModalInitialStep changes
  useEffect(() => {
    if (welcomeModalInitialStep !== null) {
      setCurrentStep(welcomeModalInitialStep);
    }
  }, [welcomeModalInitialStep]);

  const handleWelcomeClose = () => {
    try {
      localStorage.setItem('circuit-tracer-visited', 'true');
    } catch (error) {
      console.error('Error setting localStorage:', error);
    }
    // Reset the initial step to 0 for next time
    setCurrentStep(0);
    resetWelcomeModalStep();
    setIsWelcomeModalOpen(false);
  };

  const nextStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const goToStep = (stepIndex: number) => {
    setCurrentStep(stepIndex);
  };

  return (
    <Dialog open={isWelcomeModalOpen} onOpenChange={handleWelcomeClose}>
      <DialogContent className="w-screen-xl max-w-screen-xl overflow-hidden border-0 bg-white px-0 pb-0 pt-0 text-slate-700">
        <DialogHeader className="space-y-0">
          <DialogTitle className="flex w-full flex-row items-center justify-between gap-x-5 bg-white px-8 pb-4 pt-6 text-2xl font-bold leading-none tracking-normal">
            <div className="select-none whitespace-nowrap text-center leading-tight tracking-tight text-slate-700">
              Circuit Tracer
              <br />
              Guide
            </div>

            <div className="flex flex-1 flex-row gap-x-2.5 px-2 text-sm">
              <a
                href="https://youtu.be/ruLcDtr_cGo"
                target="_blank"
                rel="noopener noreferrer"
                className="flex flex-1 cursor-pointer flex-row items-center justify-center gap-x-2 whitespace-nowrap rounded-lg border border-red-600 bg-red-50 px-4 py-2.5 text-center text-[13px] font-medium text-red-600 hover:bg-red-200"
              >
                <svg className="h-7 w-7" viewBox="0 0 32 32" fill="currentColor">
                  <path d="M12.932 20.459v-8.917l7.839 4.459zM30.368 8.735c-0.354-1.301-1.354-2.307-2.625-2.663l-0.027-0.006c-3.193-0.406-6.886-0.638-10.634-0.638-0.381 0-0.761 0.002-1.14 0.007l0.058-0.001c-0.322-0.004-0.701-0.007-1.082-0.007-3.748 0-7.443 0.232-11.070 0.681l0.434-0.044c-1.297 0.363-2.297 1.368-2.644 2.643l-0.006 0.026c-0.4 2.109-0.628 4.536-0.628 7.016 0 0.088 0 0.176 0.001 0.263l-0-0.014c-0 0.074-0.001 0.162-0.001 0.25 0 2.48 0.229 4.906 0.666 7.259l-0.038-0.244c0.354 1.301 1.354 2.307 2.625 2.663l0.027 0.006c3.193 0.406 6.886 0.638 10.634 0.638 0.38 0 0.76-0.002 1.14-0.007l-0.058 0.001c0.322 0.004 0.702 0.007 1.082 0.007 3.749 0 7.443-0.232 11.070-0.681l-0.434 0.044c1.298-0.362 2.298-1.368 2.646-2.643l0.006-0.026c0.399-2.109 0.627-4.536 0.627-7.015 0-0.088-0-0.176-0.001-0.263l0 0.013c0-0.074 0.001-0.162 0.001-0.25 0-2.48-0.229-4.906-0.666-7.259l0.038 0.244z" />
                </svg>
                <div>YouTube Tutorial</div>
              </a>
              <a
                href="/graph/info"
                target="_blank"
                rel="noopener noreferrer"
                className="flex flex-1 cursor-pointer flex-row items-center justify-center gap-x-2 whitespace-nowrap rounded-lg border border-sky-600 bg-sky-50 px-4 py-2.5 text-center text-[13px] font-medium text-sky-700 hover:bg-sky-200"
              >
                <NewspaperIcon className="h-6 w-6" />
                <div>Aug 2025 Update</div>
              </a>
              <a
                href="https://github.com/safety-research/circuit-tracer"
                target="_blank"
                rel="noopener noreferrer"
                className="flex flex-1 cursor-pointer flex-row items-center justify-center gap-x-2 whitespace-nowrap rounded-lg border border-slate-600 bg-slate-50 px-4 py-2.5 text-center text-[12px] font-medium text-slate-700 hover:bg-slate-200"
              >
                <GithubIcon className="h-6 w-6" />
                <div className="leading-snug">
                  circuit-tracer
                  <br />
                  GitHub
                </div>
              </a>
              <div className="grid grid-cols-2 gap-2 gap-y-1">
                <Button
                  onClick={() => {
                    window.open('https://www.anthropic.com/research/open-source-circuit-tracing', '_blank');
                  }}
                  variant="outline"
                  size="sm"
                  className="flex h-7 flex-1 flex-row items-center justify-center gap-x-2 text-slate-500"
                >
                  <NewspaperIcon className="h-3.5 w-3.5" />
                  <div className="text-[11px]">Anthropic Blog</div>
                </Button>
                <Button
                  onClick={() => {
                    window.open(
                      'https://github.com/safety-research/circuit-tracer/blob/main/demos/circuit_tracing_tutorial.ipynb',
                      '_blank',
                    );
                  }}
                  variant="outline"
                  size="sm"
                  className="flex h-7 flex-1 flex-row items-center justify-center gap-x-2 text-slate-500"
                >
                  <NotebookIcon className="h-3.5 w-3.5" />
                  <div className="text-[11px]">Gemma 2 Notebook</div>
                </Button>

                <Button
                  onClick={() => {
                    window.open('https://transformer-circuits.pub/2025/attribution-graphs/methods.html', '_blank');
                  }}
                  variant="outline"
                  size="sm"
                  className="flex h-7 flex-1 flex-row items-center justify-center gap-x-2 text-slate-500"
                >
                  <BookOpen className="h-3.5 w-3.5" />
                  <div className="text-[11px]">Attrib Graphs Paper</div>
                </Button>

                <Button
                  onClick={() => {
                    window.open('https://transformer-circuits.pub/2025/attribution-graphs/biology.html', '_blank');
                  }}
                  variant="outline"
                  size="sm"
                  className="flex h-7 flex-1 flex-row items-center justify-center gap-x-2 text-slate-500"
                >
                  <BookOpen className="h-3.5 w-3.5" />
                  <div className="text-[11px]">Claude Haiku Study</div>
                </Button>
              </div>
            </div>
          </DialogTitle>
          <div className="flex w-full flex-1 flex-row justify-center">
            {steps.map((step, index) => (
              <button
                key={index}
                type="button"
                onClick={() => goToStep(index)}
                aria-label={`Go to step ${index + 1}: ${step.title}`}
                className={`flex flex-1 flex-col items-center justify-center gap-y-1 px-3 py-3 text-xs font-medium transition-all duration-200 focus:outline-none focus:ring-0 ${
                  index === currentStep
                    ? 'bg-slate-300 text-slate-600'
                    : 'bg-slate-200 text-slate-600 hover:bg-slate-300/80'
                } `}
              >
                <div
                  className={`whitespace-pre text-[12px] leading-none ${index === currentStep ? 'font-semibold' : ''}`}
                >
                  {step.subtitle}
                </div>

                <div
                  className={`${index === currentStep ? 'text-slate-500' : 'text-slate-400'} text-[8px] font-medium uppercase leading-none`}
                >
                  {step.title}
                </div>
              </button>
            ))}
          </div>
          <DialogDescription className="pb-4 pt-0 text-center">
            <div className="flex flex-row gap-x-0 px-8">
              <div className="mt-0 flex w-1/4 flex-col border-slate-100 pb-4 pr-0">
                <div className="flex-1 rounded-lg bg-white p-4 px-0 pb-2 pt-5">
                  {/* <div className="mb-0 text-base font-bold text-slate-600">{steps[currentStep].subtitle}</div>
                  <div className="mb-3.5 mt-1 flex items-center justify-center gap-2 text-center text-[11px] font-medium uppercase leading-none text-slate-400">
                    {steps[currentStep].title}
                  </div> */}
                  <div className="text-xs text-slate-700">{steps[currentStep].content}</div>
                </div>
                <div className="flex justify-between gap-x-2.5">
                  {currentStep === steps.length - 1 ? (
                    <Button
                      variant="default"
                      size="sm"
                      onClick={handleWelcomeClose}
                      className="flex h-[39px] flex-1 items-center gap-1 bg-slate-600 px-5 text-white shadow-none hover:bg-slate-700"
                    >
                      Close Guide
                    </Button>
                  ) : (
                    <Button
                      variant="default"
                      size="sm"
                      onClick={nextStep}
                      className="gap-1f flex h-[39px] flex-1 items-center px-5 shadow-none"
                    >
                      Next
                      <ChevronRight className="h-4 w-4" />
                    </Button>
                  )}
                </div>
              </div>
              <div className="mt-4 flex w-3/4 flex-col items-center justify-start pb-4 pl-7">
                {stepImages[currentStep].endsWith('.jpg') ? (
                  <img
                    src={stepImages[currentStep]}
                    alt={stepImageAlts[currentStep]}
                    className="max-h-[540px] min-h-[540px] max-w-full rounded-xl border-slate-200 object-contain"
                  />
                ) : (
                  <video
                    id={`video-${currentStep}`}
                    src={stepImages[currentStep]}
                    loop
                    controls
                    muted
                    className="max-h-[540px] min-h-[540px] max-w-full border-slate-200 object-contain"
                  />
                )}
              </div>
            </div>
          </DialogDescription>
        </DialogHeader>
      </DialogContent>
    </Dialog>
  );
}
