import { ControlReferenceType, GoogleGenAI, FunctionCallingConfigMode, Type } from "@google/genai";
import type { FunctionDeclaration, Content, ToolUnion, FunctionCall, Schema } from "@google/genai";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import readline from "readline/promises";
import dotenv from "dotenv";

dotenv.config();

class MCPClient {
    public mcp: Client;
    private genai: GoogleGenAI;
    private transport: StdioClientTransport | null = null;
    private tools: FunctionDeclaration[] = [];

    constructor() {
        this.genai = new GoogleGenAI({
            apiKey: process.env.GEMINI_API_KEY,
            apiVersion: 'v1alpha'
        });
        this.mcp = new Client({
            name: "mcp-client-cli",
            version: "1.0.0"
        }, {
            capabilities: {}
        });
    }

    async connectToServer(serverScriptPath: string) {
        try {
            const isJs = serverScriptPath.endsWith(".js");
            const isPy = serverScriptPath.endsWith(".py");
            const isTs = serverScriptPath.endsWith(".ts");
            if (!isJs && !isPy && !isTs) {
                throw new Error("Server script must be a .js or .py or .ts file");
            }
            
            let command: string;
            let args: string[];
            
            if (isPy) {
                command = process.platform === "win32" ? "python" : "python3";
                args = [serverScriptPath];
                console.log(command,args);
            } else {
                command = process.execPath;
                args = [serverScriptPath];
                console.log(command,args);
            }

            this.transport = new StdioClientTransport({
                command,
                args,
            });
            
            await this.mcp.connect(this.transport);

            const toolsResult = await this.mcp.listTools();
            this.tools = toolsResult.tools.map((tool) => {
                // Convert MCP tool schema to Google GenAI format
                const properties: Record<string, any> = {};
                const required: string[] = [];
                
                if (tool.inputSchema && tool.inputSchema.properties) {
                    for (const [key, value] of Object.entries(tool.inputSchema.properties)) {
                        properties[key] = this.convertSchemaProperty(value as any);
                    }
                }
                
                if (tool.inputSchema && tool.inputSchema.required) {
                    required.push(...tool.inputSchema.required);
                }

                return {
                    name: tool.name,
                    description: tool.description,
                    parameters: {
                        type: Type.OBJECT,
                        properties,
                        required
                    }
                };
            });

            console.log(
                "Connected to server with tools:",
                this.tools.map(({ name }) => name)
            );
        } catch (e) {
            console.log("Failed to connect to MCP server: ", e);
            throw e;
        }
    }

    private convertSchemaProperty(prop: any): Schema {
        // Convert Zod/JSON Schema property to Google GenAI format
        if (prop.type === 'string') {
            return {
                type: Type.STRING,
                description: prop.description
            };
        } else if (prop.type === 'number') {
            return {
                type: Type.NUMBER,
                description: prop.description
            };
        } else if (prop.type === 'boolean') {
            return {
                type: Type.BOOLEAN,
                description: prop.description
            };
        }
        
        // Default fallback
        return {
            type: Type.STRING,
            description: prop.description || ''
        };
    }

    async processQuery(query: string) {
        const contents: Content[] = [
            {
                role: "user",
                parts: [
                    {
                        text: query
                    }
                ],
            },
        ];

        const response = await this.genai.models.generateContent({
            model: "gemini-2.0-flash-001",
            contents,
            config: {
                toolConfig: {
                    functionCallingConfig: {
                        mode: FunctionCallingConfigMode.ANY,
                        allowedFunctionNames: this.tools.map(t => t.name as string),
                    }
                },
                tools: [
                    {
                        functionDeclarations: this.tools
                    }
                ]
            }
        });

        console.log("Available tools:", this.tools.map(t => t.name));
        console.log("Function calls:", response.functionCalls);

        // If no function calls, return the direct response
        if (!response.functionCalls || response.functionCalls.length === 0) {
            return response.text || "No response generated.";
        }

        // Add the model's response (with function calls) to the conversation
        contents.push({
            role: "model",
            parts: response.functionCalls.map(call => ({
                functionCall: {
                    name: call.name,
                    args: call.args || {}
                }
            }))
        });

        // Process each function call and add responses
        const functionResponseParts = [];
        
        for (const functionCall of response.functionCalls) {
            const toolName = functionCall.name;
            const toolArgs = functionCall.args || {};

            console.log(`Calling tool: ${toolName} with args:`, toolArgs);

            try {
                const result = await this.mcp.callTool({
                    name: toolName as string,
                    arguments: toolArgs,
                }); 

                // Extract text content from MCP result
                let resultText = "";
                if (result.content && Array.isArray(result.content)) {
                    resultText = result.content
                        .filter(item => item.type === "text")
                        .map(item => item.text)
                        .join("\n");
                } else if (typeof result.content === "string") {
                    resultText = result.content;
                }

                functionResponseParts.push({
                    functionResponse: {
                        name: toolName,
                        response: {
                            result: resultText
                        }
                    }
                });

            } catch (error) {
                console.error(`Error calling tool ${toolName}:`, error);
                functionResponseParts.push({
                    functionResponse: {
                        name: toolName,
                        response: {
                            error: `Error calling tool: ${error}`
                        }
                    }
                });
            }
        }

        // Add function responses to conversation
        contents.push({
            role: "user",
            parts: functionResponseParts
        });

        // Generate final response with tool results
        const finalResponse = await this.genai.models.generateContent({
            model: "gemini-2.0-flash-001",
            contents,
        });

        return finalResponse.text || "No final response generated.";
    }

    async chatLoop() {
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout,
        });

        try {
            console.log("\nMCP Client Started!");
            console.log("Type your queries or 'quit' to exit.");

            while (true) {
                const message = await rl.question("\nQuery: ");
                if (message.toLowerCase() === "quit") {
                    break;
                }
                
                try {
                    const response = await this.processQuery(message);
                    console.log("\n" + response);
                } catch (error) {
                    console.error("Error processing query:", error);
                }
            }
        } 
        finally {
            rl.close();
        }
    }

    async cleanup() {
        if (this.transport) {
            await this.transport.close();
        }
        await this.mcp.close();
    }
}

async function main() {
    if (process.argv.length < 3) {
        console.log("Usage: node index.ts <path_to_server_script>");
        return;
    }
    
    const mcpClient = new MCPClient();
    try {
        await mcpClient.connectToServer(process.argv[2] as string);
        await mcpClient.chatLoop();
    } finally {
        await mcpClient.cleanup();
        process.exit(0);
    }
}

main().catch(error => {
    console.error("Fatal error:", error);
    process.exit(1);
});