import { PDFLoader  } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { ChromaClient } from "chromadb";
export class BibleIngestion {
    async ingest_data() {
        try {
           const client = new ChromaClient();

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

            // 3. Create vector store, embeddings and add documents
            console.log("Creating vector store and embedding documents...");
            const collection = await client.getOrCreateCollection({
                name: "bible_collection",
            });

            const model = new HuggingFaceTransformersEmbeddings({ 
                model: 'sentence-transformers/all-MiniLM-L6-v2' 
            });
            console.log("Model loaded successfully!");
            
            for (const[index, doc] of splitDocs.entries()) {
                const vector = await model.embedQuery(doc.pageContent);
                await collection.add({
                    ids: [`doc-${index}`],
                    documents: [doc.pageContent],
                    embeddings: [vector],
                    metadatas: [{
                        source: "kjv.pdf",
                        chunk_id: index
                    }]
                })
            }
            
            console.log("Data successfully embedded and stored in vector database!");
            return { success: true, documentsProcessed: splitDocs.length };

        } catch (error) {
            console.error("Error during ingestion:", error);
            throw error;
        }
    }

    async vector_search(query: string, topK = 10) {
        try {
           const client = new ChromaClient();
            //1. Retrieve the ChromaDb
            const collection = await client.getOrCreateCollection({
                name: "bible_collection",
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

  