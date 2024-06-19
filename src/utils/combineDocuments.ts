// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function combineDocuments(docs: any[]){
    return docs.map((doc)=>doc.pageContent).join('\n\n')
}