import { createRAG, RAGConfig as OriginalRAGConfig } from 'rag-module/dist/index.js';

interface RAGConfig extends OriginalRAGConfig {
    prompts: {
        classification: string;
        response: string;
    };
}

const config: RAGConfig = {
    googleApiKey: 'your-google-api-key',
    qdrantUrl: 'your-qdrant-url',
    qdrantApiKey: 'your-qdrant-api-key',
    groqApiKey: 'your-groq-api-key',
    collectionName: 'custom-collection',
    fallbackResponse: 'Sorry, I could not find relevant information.',
    prompts: {
        // Custom classification prompt
        classification: `Analyze this query and categorize it as either 'technical', 'historical', or 'general': {query}. 
                        Reply with just one word: technical, historical, or general.`,
        
        // Custom response prompt
        response: `As a knowledgeable assistant, analyze this context: '{context}'
                and provide a detailed answer to this question: '{query}'.
                Include relevant quotes from the context when appropriate.
                If the context doesn't contain relevant information, explain why.`
    }
};

const rag = createRAG(config);

// Use the RAG implementation
async function main() {
  try {
    const result = await rag.processQuery(
      "context...",
      "query..."
    );
    console.log('Response:', result.response);
    console.log('Category:', result.category);
  } catch (error) {
    console.error('Error:', error);
  }
}

main().catch(console.error);