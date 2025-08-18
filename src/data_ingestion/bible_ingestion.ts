/* This is where the indexing process takes place. The data will be loaded, split into chunks and stored in a vector database */
import { PDFLoader  } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb"
import { MongoClient } from "mongodb";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";




export class BibleIngestion {
    async ingest_data() {
        // Validate environment variables
        const config = {
            uri: process.env.MONGODB_ATLAS_URI,
            dbName: process.env.MONGODB_ATLAS_DB_NAME,
            collectionName: process.env.MONGODB_ATLAS_COLLECTION1_NAME
        };

        if (!config.uri || !config.dbName || !config.collectionName) {
            throw new Error("Missing required MongoDB environment variables");
        }

        let client : MongoClient | null = null;

        
        try {
           
            // 1. Loading the PDF file
            console.log("Loading Bible PDF file...");
            const pdfPath = "./data/kjv.pdf"; // PDF file path
            const loader = new PDFLoader(pdfPath);
            const docs = await loader.load();

            console.log(`PDF loaded successfully! Documents: ${docs.length}`);

            // Log first few characters to verify content
            if (docs.length > 0) {
                console.log("Sample content:", docs[0].pageContent.substring(0, 200));
            }

            // 2. Spilting the data into chunks
            console.log("Splitting data into chunks...");    
            const textSplitter = new RecursiveCharacterTextSplitter({
                chunkSize: 1000, // Maximum size of each chunk
                chunkOverlap: 100, // Overlap between consecutive chunks to maintain context
            });

            const splitDocs = await textSplitter.splitDocuments(docs);
            // 'splitDocs' will be an array of Document objects, each representing a chunk of the original PDF content.

            console.log(`Data split successfully! Chunks created: ${splitDocs.length}`);

            // 3. Initialize MongoDB connection
            console.log("Connecting to MongoDB Atlas...");
            client = new MongoClient(config.uri);
            await client.connect();
            
            const collection = client
                .db(config.dbName)
                .collection(config.collectionName);

            // 4. Initailize the embeddings model
            console.log("Initializing embeddings models...");
            const embeddings = new HuggingFaceTransformersEmbeddings({ 
                model: 'sentence-transformers/all-MiniLM-L6-v2' 
            })

            // 5. Create vector store and add documents
            console.log("Creating vector store and embedding documents...");
            const vectorStore = new MongoDBAtlasVectorSearch(embeddings, {
                collection: collection,
                indexName: "vector_index", // ensure this matches your Atlas index name
                textKey: "text",
                embeddingKey: "embedding",
            });

            await vectorStore.addDocuments(splitDocs);

            console.log("Data successfully embedded and stored in vector database!");
            return { success: true, documentsProcessed: splitDocs.length };

        } catch (error) {
            console.error("Error during ingestion:", error);
            throw error;
        } finally{
            // close the MongoDB connection
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
            collectionName: process.env.MONGODB_ATLAS_COLLECTION1_NAME
        };

        if (!config.uri || !config.dbName || !config.collectionName) {
            throw new Error("Missing required MongoDB environment variables for vector search");
        }
        let client = null;

        try {
            // 1. Connect to MongoDB
            console.log("Connecting to MongoDB for bible vector search...");
            client = new MongoClient(config.uri);
            await client.connect();
            
            const collection = client
                .db(config.dbName)
                .collection(config.collectionName);

            // 2. Generate embedding for the search query
            console.log("Generating embedding for query:", query);
            const embeddings = new HuggingFaceTransformersEmbeddings({ 
                model: 'sentence-transformers/all-MiniLM-L6-v2' 
            });
            
            const queryEmbedding = await embeddings.embedQuery(query);
            console.log("Query embedding generated, dimension:", queryEmbedding.length);

            // 3. MongoDB Aggregation Pipeline for Vector Search
            const pipeline = [
                {
                    $vectorSearch: {
                        index: "vector_index", // Must match your Atlas Search index name
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
                source: "Bible"
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

  