import { ModelInfo, OpenAiCompatibleModelInfo, openAiModelInfoSaneDefaults } from "@shared/api"
import OpenAI from "openai"
import type { ChatCompletionReasoningEffort, ChatCompletionTool } from "openai/resources/chat/completions"
import { ClineStorageMessage } from "@/shared/messages/content"
import { fetch } from "@/shared/net"
import { ApiHandler, CommonApiHandlerOptions } from "../index"
import { withRetry } from "../retry"
import { convertToOpenAiMessages } from "../transform/openai-format"
import { ApiStream } from "../transform/stream"
import { getOpenAIToolParams, ToolCallProcessor } from "../transform/tool-call-processor"

interface GaussHandlerOptions extends CommonApiHandlerOptions {
	gaussApiKey?: string
	gaussClientKey?: string
	gaussBaseUrl?: string
	gaussModelId?: string
	gaussModelInfo?: OpenAiCompatibleModelInfo
	reasoningEffort?: string
}

export class GaussHandler implements ApiHandler {
	private options: GaussHandlerOptions
	private client: OpenAI | undefined

	constructor(options: GaussHandlerOptions) {
		this.options = options
	}

	private ensureClient(): OpenAI {
		if (!this.client) {
			if (!this.options.gaussApiKey) {
				throw new Error("Gauss API key is required")
			}
			if (!this.options.gaussClientKey) {
				throw new Error("Gauss Client key is required")
			}
			try {
				this.client = new OpenAI({
					baseURL: this.options.gaussBaseUrl || "https://api.gauss.ai/v1",
					apiKey: this.options.gaussApiKey,
					defaultHeaders: {
						"X-Client-Key": this.options.gaussClientKey,
					},
					fetch, // Use configured fetch with proxy support
				})
			} catch (error: any) {
				throw new Error(`Error creating Gauss client: ${error.message}`)
			}
		}
		return this.client
	}

	@withRetry()
	async *createMessage(systemPrompt: string, messages: ClineStorageMessage[], tools?: ChatCompletionTool[]): ApiStream {
		const client = this.ensureClient()
		const modelId = this.options.gaussModelId ?? ""

		const openAiMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [
			{ role: "system", content: systemPrompt },
			...convertToOpenAiMessages(messages),
		]

		let temperature: number | undefined
		if (this.options.gaussModelInfo?.temperature !== undefined) {
			const tempValue = Number(this.options.gaussModelInfo.temperature)
			temperature = tempValue === 0 ? undefined : tempValue
		} else {
			temperature = openAiModelInfoSaneDefaults.temperature
		}

		let maxTokens: number | undefined
		if (this.options.gaussModelInfo?.maxTokens && this.options.gaussModelInfo.maxTokens > 0) {
			maxTokens = Number(this.options.gaussModelInfo.maxTokens)
		} else {
			maxTokens = undefined
		}

		let reasoningEffort: ChatCompletionReasoningEffort | undefined
		if (this.options.gaussModelInfo?.supportsReasoningEffort) {
			reasoningEffort = (this.options.reasoningEffort as ChatCompletionReasoningEffort) || undefined
		}

		const stream = await client.chat.completions.create({
			model: modelId,
			messages: openAiMessages,
			temperature,
			max_tokens: maxTokens,
			reasoning_effort: reasoningEffort,
			stream: true,
			stream_options: { include_usage: true },
			...getOpenAIToolParams(tools),
		})

		const toolCallProcessor = new ToolCallProcessor()

		for await (const chunk of stream) {
			const delta = chunk.choices[0]?.delta
			if (delta?.content) {
				yield {
					type: "text",
					text: delta.content,
				}
			}

			if (delta && "reasoning_content" in delta && delta.reasoning_content) {
				yield {
					type: "reasoning",
					reasoning: (delta.reasoning_content as string | undefined) || "",
				}
			}

			if (delta?.tool_calls) {
				yield* toolCallProcessor.processToolCallDeltas(delta.tool_calls)
			}

			if (chunk.usage) {
				yield {
					type: "usage",
					inputTokens: chunk.usage.prompt_tokens || 0,
					outputTokens: chunk.usage.completion_tokens || 0,
					cacheReadTokens: chunk.usage.prompt_tokens_details?.cached_tokens || 0,
					// @ts-ignore-next-line
					cacheWriteTokens: chunk.usage.prompt_cache_miss_tokens || 0,
				}
			}
		}
	}

	getModel(): { id: string; info: ModelInfo } {
		return {
			id: this.options.gaussModelId ?? "",
			info: this.options.gaussModelInfo ?? openAiModelInfoSaneDefaults,
		}
	}
}
