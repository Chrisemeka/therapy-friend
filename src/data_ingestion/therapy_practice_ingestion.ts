/* This is where the indexing process takes place. The data will be loaded, split into chunks and stored in a vector database */
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb"
import { MongoClient } from "mongodb";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import path from "path";

export class TherapyDataIngestion {
    async ingest_data() {
        // Validate environment variables
        const config = {
            uri: process.env.MONGODB_ATLAS_URI,
            dbName: process.env.MONGODB_ATLAS_DB_NAME,
            collectionName: process.env.MONGODB_ATLAS_COLLECTION2_NAME
        };

        if (!config.uri || !config.dbName || !config.collectionName) {
            throw new Error("Missing required MongoDB environment variables");
        }

        let client : MongoClient | null = null;

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

            // 3. Initialize MongoDB connection
            console.log("Connecting to MongoDB Atlas...");
            client = new MongoClient(config.uri);
            await client.connect();
            
            const collection = client
                .db(config.dbName)
                .collection(config.collectionName);
                
            // 4. Initialize the embedding model
            console.log("Initializing embedding model...");
            const embeddings = new HuggingFaceTransformersEmbeddings({
                model: 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
            });

            // 5. Create vector store and add documents
            console.log("Creating vector store and embedding documents...");
            const vectorStore = new MongoDBAtlasVectorSearch(embeddings, {
                collection: collection,
                indexName: "cbt_vector_index",
                textKey: "text",
                embeddingKey: "embedding",
            });

            await vectorStore.addDocuments(splitDocs);

            return { 
                success: true, 
                documentsProcessed: docs.length,
                chunksCreated: splitDocs.length,
            };
        } catch (error) {
            console.error("Error during therapy data ingestion:", error);
            throw error;
        } finally{
            if (client) {
                console.log("Closing MongoDB connection...");
                await client.close();
            }
        }
    }

    async vector_search(query: string, topK = 10) {
        const config = {
            uri: process.env.MONGODB_ATLAS_URI,
            dbName: process.env.MONGODB_ATLAS_DB_NAME,
            collectionName: process.env.MONGODB_ATLAS_COLLECTION2_NAME
        };

        if (!config.uri || !config.dbName || !config.collectionName) {
            throw new Error("Missing required MongoDB environment variables for vector search in therapy DB");
        }
        let client = null;

        try {
            // 1. Connect to MongoDB
            console.log("Connecting to MongoDB for therapy vector search...");
            client = new MongoClient(config.uri);
            await client.connect();
            
            const collection = client
                .db(config.dbName)
                .collection(config.collectionName);

            // 2. Generate embedding for the search query
            console.log("Generating embedding for query:", query);
            const embeddings = new HuggingFaceTransformersEmbeddings({ 
                model: 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1' 
            });
            
            const queryEmbedding = await embeddings.embedQuery(query);
            console.log("Query embedding generated, dimension:", queryEmbedding.length);

            // 3. MongoDB Aggregation Pipeline for Vector Search
            const pipeline = [
                {
                    $vectorSearch: {
                        index: "cbt_vector_index", // Must match your Atlas Search index name
                        path: "embedding",     // Field containing the embeddings
                        queryVector: queryEmbedding, // The query embedding array
                        numCandidates: topK * 10, // Number of candidates to consider
                        limit: topK            // Final number of results to return
                    }
                },
                {
                    $project: {
                        _id: 1,
                        text: 1,               // Return the text content
                        score: { $meta: "vectorSearchScore" } // Similarity score
                    }
                }
            ];

            // 4. Execute the aggregation pipeline
            console.log("Executing vector search...");
            const results = await collection.aggregate(pipeline).toArray();

            console.log(`Vector search completed! Found ${results.length} results`);

            // 5. Format results
            return results.map(result => ({
                content: result.text,
                score: result.score,
                documentId: result._id.toString(),
                source: "Therapy"
            }));

        } catch (error) {
            console.error("Error during vector search:", error);
            throw error;
        } finally {
            if (client) {
                console.log("Closing MongoDB connection...");
                await client.close();
            }
        }
    }
}
