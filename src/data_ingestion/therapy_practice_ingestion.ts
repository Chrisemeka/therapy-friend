import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { Chroma } from "@langchain/community/vectorstores/chroma";

import path from "path";

export class TherapyDataIngestion {
    async ingest_data() {
        try {
            // 1. Loading multiple text files from CBT data directory
            console.log("Loading Therapy data files...");

            const loader = new DirectoryLoader(
                "./data/cbt_content/", // Path to the directory
                {
                    ".txt": (path) => new TextLoader(path), // Load .txt files in the directory with TextLoader
                }
            );
            const docs = await loader.load();

            console.log(`Text documents loaded successfully! Documents: ${docs.length}`);


            // Add more metadata information to the documents for better categorization
            const docsWithMetadata = docs.map((doc) => {
                const filename = path.basename(doc.metadata.source);
                const category = filename.replace('.txt', '').replace('_', ' ');

                return {
                    ...doc,
                    metadata: {
                        ...doc.metadata,
                        category: category,
                        source_type: 'cbt_technique'
                    }
                };
            });

            //2. Spliting the data into chunks
            console.log("Splitting data into chunks...");    
            const textSplitter = new RecursiveCharacterTextSplitter({
                chunkSize: 500, 
                chunkOverlap: 50,
                separators: ["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""],
            });

            const splitDocs = await textSplitter.splitDocuments(docsWithMetadata);
            console.log(`Data split successfully! Chunks created: ${splitDocs.length}`);
                
            // 3. Initialize the embedding model
            console.log("Initializing embedding model...");
            const embeddings = new HuggingFaceTransformersEmbeddings({
                model: 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
            });

            // 4 . Create vector store and add documents
            console.log("Creating vector store and embedding documents...");
            const vectorStore = new Chroma(embeddings, {
                collectionName: "therapy-collection",
            });
            const ids = splitDocs.map((_, index) => `doc-${index}`);
            await vectorStore.addDocuments(splitDocs, {ids});

            console.log("Data successfully embedded and stored in vector database!");
            return { success: true, documentsProcessed: splitDocs.length };

        } catch (error) {
            console.error("Error during therapy data ingestion:", error);
            throw error;
        }
    }

    async vector_search(query: string, topK = 10) {
        try {
            // 1. Generate embedding for the search query
            console.log("Generating embedding for query:", query);
            const embeddings = new HuggingFaceTransformersEmbeddings({ 
                model: 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1' 
            });
            
            const queryEmbedding = await embeddings.embedQuery(query);
            console.log("Query embedding generated, dimension:", queryEmbedding.length);

            //2. Retrieve the ChromaDb
            const vectorStore = new Chroma(embeddings, {
                collectionName: "therapy-collection",
            });

            const results = await vectorStore.similaritySearch(query, topK);
            return results

        } catch (error) {
            console.error("Error during vector search:", error);
            throw error;
        }
        
    }
}
