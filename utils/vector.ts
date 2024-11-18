import { SupabaseFilterRPCCall, SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { createClient } from "@supabase/supabase-js";
import { Document } from "langchain/dist/document";
import { OPENAI_TYPE, SUPABASE_KEY, SUPABASE_URL } from "@/utils/app/const";
import { getEmbeddings } from "@/utils/embeddings";
import { KeyConfiguration, ModelType } from "@/types";

const client = createClient(SUPABASE_URL!, SUPABASE_KEY!);

const sanitizeContent = (content) => {
  if (!content) return '';

  const sanitizedContent = content
    .replace(/\\u([0-9a-fA-F]{4})/g, '') // Remove escape sequences like \uXXXX
    .replace(/[^\x20-\x7E]/g, ' '); // Replace non-ASCII characters with spaces

  return sanitizedContent.trim();
};

export const getVectorStore = async (keyConfiguration: KeyConfiguration, texts: string[], metadata: object) => {
  return await SupabaseVectorStore.fromTexts(
    texts.map(sanitizeContent), // Apply sanitization
    metadata,
    await getEmbeddings(keyConfiguration),
    {
      client,
      tableName: "documents",
      queryName: "match_documents",
    }
  );
};

export const saveEmbeddings = async (keyConfiguration: KeyConfiguration, documents: Document[]) => {
  const supabaseVectorStore = new SupabaseVectorStore(await getEmbeddings(keyConfiguration), {
    client,
    tableName: "documents",
    queryName: "match_documents",
  });

  // Sanitize document content before saving
  const sanitizedDocuments = documents.map((doc) => ({
    ...doc,
    pageContent: sanitizeContent(doc.pageContent),
  }));

  // Handle Azure API type
  if (keyConfiguration.apiType === ModelType.AZURE_OPENAI) {
    for (const doc of sanitizedDocuments) {
      await supabaseVectorStore.addDocuments([doc]);
    }
  } else {
    await supabaseVectorStore.addDocuments(sanitizedDocuments);
  }
};

export const getExistingVectorStore = async (keyConfiguration: KeyConfiguration, fileName: string) => {
  const fileNameFilter: SupabaseFilterRPCCall = (rpc) =>
    rpc.filter("metadata->>file_name", "eq", fileName);

  return await SupabaseVectorStore.fromExistingIndex(await getEmbeddings(keyConfiguration), {
    client,
    tableName: "documents",
    queryName: "match_documents",
    filter: fileNameFilter,
  });
};
