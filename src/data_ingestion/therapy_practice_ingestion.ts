import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { ChromaClient } from "chromadb";


import path from "path";

export class TherapyDataIngestion {
    async ingest_data() {
        const client = new ChromaClient();

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

            // 3. Create vector store and add documents
            console.log("Creating vector store and embedding documents...");
            const collection = await client.getOrCreateCollection({
                name: "therapy_collection",
            });

            const model = new HuggingFaceTransformersEmbeddings({ 
                model: 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1' 
            });
            console.log("Model loaded successfully!");
            
            for (const[index, doc] of splitDocs.entries()) {
                const vector = await model.embedQuery(doc.pageContent);
                await collection.add({
                    ids: [`doc-${index}`],
                    documents: [doc.pageContent],
                    embeddings: [vector],
                    metadatas: [{
                        source: "cbt_content.pdf",
                        chunk_id: index
                    }]
                })
            }

            console.log("Data successfully embedded and stored in vector database!");
            return { success: true, documentsProcessed: splitDocs.length };

        } catch (error) {
            console.error("Error during therapy data ingestion:", error);
            throw error;
        }
    }

    async vector_search(query: string, topK = 10) {
        try {
            const client = new ChromaClient();
            //1. Retrieve the ChromaDb
            const collection = await client.getOrCreateCollection({
                name: "therapy_collection",
            });

            const results = await collection.query({
                queryTexts: [query],
                nResults: topK,
            })
            return results
        } catch (error) {
            console.error("Error during vector search:", error);
            throw error;
        }
        
    }
}
