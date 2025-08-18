import { MongoClient } from "mongodb";

export const checkBibleVectorDB = async () => {
    try {
        const uri = process.env.MONGODB_ATLAS_URI;
        const dbName = process.env.MONGODB_ATLAS_DB_NAME;
        const collectionName = process.env.MONGODB_ATLAS_COLLECTION1_NAME;

        if (!uri || !dbName || !collectionName) {
            throw new Error("Missing required MongoDB environment variables bible collection");
        }

        const client = new MongoClient(uri);
        await client.connect();
        const db = client.db(dbName);
        const collection = db.collection(collectionName);

        const count = await collection.estimatedDocumentCount();
        return count > 0;
    } catch (error) {
        return false
    }
};


export const checkTherapyVectorDB = async () => {
    try {
        const uri = process.env.MONGODB_ATLAS_URI;
        const dbName = process.env.MONGODB_ATLAS_DB_NAME;
        const collectionName = process.env.MONGODB_ATLAS_COLLECTION2_NAME;

        if (!uri || !dbName || !collectionName) {
            throw new Error("Missing required MongoDB environment variables for therapy collection");
        }

        const client = new MongoClient(uri);
        await client.connect();
        const db = client.db(dbName);
        const collection = db.collection(collectionName);

        const count = await collection.estimatedDocumentCount();
        return count > 0;
    } catch (error) {
        return false
    }
};