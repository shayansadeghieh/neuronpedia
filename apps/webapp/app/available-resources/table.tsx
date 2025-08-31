'use client';

import CustomTooltip from '@/components/custom-tooltip';
import { Button } from '@/components/shadcn/button';
import { CONTACT_EMAIL_ADDRESS, NEXT_PUBLIC_URL } from '@/lib/env';
import { getLayerNumFromSource } from '@/lib/utils/source';
import { ModelWithRelations } from '@/prisma/generated/zod';
import { InfoIcon } from 'lucide-react';
import { useState } from 'react';

export default function AvailableResourcesTable({ models }: { models: ModelWithRelations[] }) {
  const [filterToOnlyInferenceEnabled, setFilterToOnlyInferenceEnabled] = useState(false);

  return (
    <div className="flex flex-col gap-2 pb-10">
      <div className="flex w-full items-center justify-center gap-2">
        <label
          htmlFor="inference-filter"
          className="flex items-center gap-1.5 text-[11px] font-medium uppercase text-slate-500"
        >
          <input
            type="checkbox"
            id="inference-filter"
            name="inference-filter"
            checked={filterToOnlyInferenceEnabled}
            onChange={(e) => setFilterToOnlyInferenceEnabled(e.target.checked)}
            className="h-4 w-4 rounded border-slate-300 text-sky-700 focus:ring-sky-500"
          />
          Show only inference-enabled sources
        </label>
      </div>
      <table className="min-w-full">
        <thead className="sticky top-0 z-10 bg-slate-100">
          <tr>
            <th className="px-4 py-2 text-left">Model ID</th>
            {/* <th className="px-4 py-2 text-left">Source Set</th> */}
            <th className="px-4 py-2 text-left">Source ID</th>
            <th className="flex flex-row items-center gap-x-2 px-4 py-2 text-left">
              Inference Enabled{' '}
              <CustomTooltip wide trigger={<InfoIcon className="h-4 w-4 text-slate-600" />}>
                <div className="mb-2 text-sm font-bold">Inference Enabled</div>
                Whether or not you can call inference API methods such as:
                <br />
                - Search via Inference
                <br />
                - Search TopK by Token
                <br />
                - Steering
                <br />
                - Activation Testing
                <br />- Etc
              </CustomTooltip>
            </th>
          </tr>
        </thead>
        <tbody>
          {models.map((model) => {
            let rowIndex = 0;

            return model.sourceSets
              .sort((a, b) => a.name.localeCompare(b.name))
              .flatMap((sourceSet) =>
                sourceSet.sources
                  .sort((a, b) => getLayerNumFromSource(a.id) - getLayerNumFromSource(b.id))
                  .map((source) => {
                    const isEven = rowIndex % 2 === 0;
                    rowIndex += 1;

                    return (
                      <tr
                        key={`${model.id}-${source.id}`}
                        className={`hover:bg-sky-100 ${isEven ? 'bg-slate-50' : 'bg-white'} ${filterToOnlyInferenceEnabled && !source.inferenceEnabled ? 'hidden' : ''}`}
                      >
                        <td className="w-1/3 px-4 py-1">
                          <a
                            target="_blank"
                            rel="noopener noreferrer"
                            href={`${NEXT_PUBLIC_URL}/${model.id}`}
                            className="font-mono text-sky-700 hover:underline"
                          >
                            {model.id}
                          </a>
                        </td>
                        {/* <td className="px-4 py-1">
                          <a
                            href={`${NEXT_PUBLIC_URL}/${model.id}/${sourceSet.name}`}
                            className="text-sky-600 hover:underline"
                          >
                            {sourceSet.name}
                          </a>
                        </td> */}
                        <td className="w-1/3 px-4 py-1">
                          <a
                            target="_blank"
                            rel="noopener noreferrer"
                            href={`${NEXT_PUBLIC_URL}/${model.id}/${source.id}`}
                            className="font-mono text-sky-700 hover:underline"
                          >
                            {source.id}
                          </a>
                        </td>
                        <td className="w-1/3 px-4 py-1">
                          {source.inferenceEnabled ? (
                            '✅'
                          ) : (
                            <div className="flex flex-row items-center gap-x-2">
                              ❌{' '}
                              <Button
                                variant="outline"
                                size="xs"
                                onClick={() => {
                                  window.open(
                                    `mailto:${CONTACT_EMAIL_ADDRESS}?subject=Request%20Inference%20for%20${model.id}%20${source.id}&body=I'd%20like%20to%20request%20inference%20for%20the%20${model.id}%20${source.id}%20source.%20Thanks!`,
                                    '_blank',
                                  );
                                }}
                              >
                                Request
                              </Button>
                            </div>
                          )}
                        </td>
                      </tr>
                    );
                  }),
              );
          })}
        </tbody>
      </table>
    </div>
  );
}
