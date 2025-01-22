import { QdrantClient } from '@qdrant/js-client-rest';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import Groq from 'groq-sdk';

interface RAGConfig {
  googleApiKey: string;
  qdrantUrl: string;
  qdrantApiKey: string;
  groqApiKey: string;
  collectionName?: string;
  fallbackResponse?: string;
  prompts?: {
    classification?: string;
    response?: string;
  };
}

interface RAGResponse {
  response: string;
  category: string;
}

export class RAGImplementation {
  private qdrantClient: QdrantClient;
  private embeddings: GoogleGenerativeAIEmbeddings;
  private groqClient: Groq;
  private collectionName: string;
  private fallbackResponse: string;
  private prompts: {
    classification: string;
    response: string;
  };

  constructor(config: RAGConfig) {
    if (!config.googleApiKey || !config.qdrantUrl || !config.qdrantApiKey || !config.groqApiKey) {
      throw new Error('Required configuration parameters are missing');
    }

    this.qdrantClient = new QdrantClient({
      url: config.qdrantUrl,
      apiKey: config.qdrantApiKey,
    });

    this.embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: config.googleApiKey,
      model: "text-embedding-004",
    });

    this.groqClient = new Groq({
      apiKey: config.groqApiKey,
    });

    this.collectionName = config.collectionName || 'default_collection';
    this.fallbackResponse = config.fallbackResponse || 'I could not find relevant information for your query.';

    // Default prompts
    this.prompts = {
      classification: config.prompts?.classification || 
        `I am providing you a query, based on the query your work is detect whether that is related to marks, events or general information. \n\nQuery: {query} \n\nPlease type marks, events or general based on the query. \n\nPlease type only one of the three words and dont type any other text. \n\nIf you are not sure, you can type general. Write your lowercase only, never put anything in uppercase`,
      response: config.prompts?.response ||
        `You are a helpful assistant. Based on this context: "{context}", please answer this question: "{query}". If you cannot find a relevant answer in the context, please say so.`
    };
  }

  private async classifyQuery(query: string): Promise<string> {
    const prompt = this.prompts.classification.replace('{query}', query);
    
    const completion = await this.groqClient.chat.completions.create({
      messages: [{ role: 'user', content: prompt }],
      model: 'llama3-8b-8192',
    });

    return completion.choices[0]?.message?.content || 'general';
  }

  private async ensureCollection(vectorSize: number): Promise<void> {
    try {
      const collection = await this.qdrantClient.getCollection(this.collectionName);
      
      if (!collection.config.params.vectors || collection.config.params.vectors.size !== vectorSize) {
        throw new Error(`Vector size mismatch. Expected: ${vectorSize}`);
      }
    } catch (error: any) {
      if (error.status === 404) {
        await this.qdrantClient.createCollection(this.collectionName, {
          vectors: {
            size: vectorSize,
            distance: 'Cosine',
          },
          optimizers_config: {
            default_segment_number: 2,
          },
          replication_factor: 2,
          write_consistency_factor: 1,
        });
      } else {
        throw error;
      }
    }
  }

  private async createVectorStore(text: string): Promise<void> {
    if (!text?.trim()) {
      throw new Error('No text provided for vector store creation');
    }

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const documentTexts = await textSplitter.splitText(text);
    const embeds = await Promise.all(
      documentTexts.map(text => this.embeddings.embedQuery(text))
    );

    await this.ensureCollection(embeds[0].length);

    const points = documentTexts.map((text, i) => ({
      id: Date.now() + i,
      vector: Array.from(embeds[i]),
      payload: {
        text: text.trim(),
        timestamp: new Date().toISOString(),
        chunkIndex: i
      }
    }));

    const batchSize = 10;
    for (let i = 0; i < points.length; i += batchSize) {
      const batch = points.slice(i, i + batchSize);
      await this.qdrantClient.upsert(this.collectionName, {
        points: batch,
        wait: true
      });
    }
  }

  private async getBotResponse(userInput: string): Promise<string> {
    const queryEmbed = await this.embeddings.embedQuery(userInput);
    
    const searchResults = await this.qdrantClient.search(this.collectionName, {
      vector: Array.from(queryEmbed),
      limit: 3,
    });

    if (!searchResults.length) {
      return this.fallbackResponse;
    }

    const context = searchResults
      .map(hit => hit.payload?.text || '')
      .join(" ");

    const prompt = this.prompts.response
      .replace('{context}', context)
      .replace('{query}', userInput);

    const completion = await this.groqClient.chat.completions.create({
      messages: [{ role: 'user', content: prompt }],
      model: 'llama3-8b-8192',
    });

    return completion.choices[0]?.message?.content || this.fallbackResponse;
  }

  public async processQuery(contextText: string, query: string): Promise<RAGResponse> {
    try {
      const category = await this.classifyQuery(query);
      await this.createVectorStore(contextText);
      const response = await this.getBotResponse(query);

      return {
        response,
        category
      };
    } catch (error) {
      console.error('Error processing query:', error);
      throw error;
    }
  }

  // Method to update prompts after initialization
  public updatePrompts(newPrompts: Partial<typeof this.prompts>): void {
    this.prompts = {
      ...this.prompts,
      ...newPrompts
    };
  }
}

// Export types for TypeScript users
export type { RAGConfig, RAGResponse };

// Export a factory function for easier instantiation
export function createRAG(config: RAGConfig): RAGImplementation {
  return new RAGImplementation(config);
}