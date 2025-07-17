import { withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

/**
@swagger
{
  "/api/explanation/export": {
    "get": {
      "tags": [
        "Exports"
      ],
      "summary": "Export Explanations, Activations, and Other Data",
      "security": [{
          "apiKey": []
      }],
      "description": "[API MOVED] Find explanation and other exports at https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/ instead.",
      "parameters": [
      ],
      "responses": {
      }
    }
  }
}
*/
export const GET = withOptionalUser(async () =>
  NextResponse.json(
    {
      message:
        'Moved to https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/ \nContact support@neuronpedia.org if you cannot find what you need there.',
      newUrl: 'https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/',
    },
    { status: 400 },
  ),
);
