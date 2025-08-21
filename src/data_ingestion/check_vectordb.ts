import { ChromaClient } from "chromadb";


export const checkBibleVectorDB = async () => {
    try {
        const client = new ChromaClient();
        const collection = await client.getCollection({
            name: "bible-collection"
        });
        const count = await collection.count();
        return count > 0;
    } catch (error) {
        return false
    }
};


export const checkTherapyVectorDB = async () => {
    try {
        const client = new ChromaClient();
        const collection = await client.getCollection({
            name: "therapy-collection"
        });
        const count = await collection.count();
        return count > 0;
    } catch (error) {
        return false
    }
};