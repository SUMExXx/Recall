package com.example.recall

import android.content.Context
import android.util.Log
import com.geniex.sdk.GenieXSdk
import com.geniex.sdk.LlmWrapper
import com.geniex.sdk.ModelManagerWrapper
import com.geniex.sdk.VlmWrapper
import com.geniex.sdk.bean.ChatMessage
import com.geniex.sdk.bean.ComputeUnitValue
import com.geniex.sdk.bean.GenerationConfig
import com.geniex.sdk.bean.HubSource
import com.geniex.sdk.bean.LlmCreateInput
import com.geniex.sdk.bean.LlmStreamResult
import com.geniex.sdk.bean.ModelConfig
import com.geniex.sdk.bean.ModelPullInput
import com.geniex.sdk.bean.ModelType
import com.geniex.sdk.bean.SamplerConfig
import com.geniex.sdk.bean.VlmChatMessage
import com.geniex.sdk.bean.VlmContent
import com.geniex.sdk.bean.VlmCreateInput
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.runBlocking
import java.io.File
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine

/** Thrown when no GenieX model is available on the device yet. */
class ModelUnavailableException(message: String) : Exception(message)

/**
 * Runs a generative model on the Snapdragon NPU via Qualcomm's GenieX SDK.
 * Handles both text LLMs ([LlmWrapper]) and vision-language models
 * ([VlmWrapper]); the memory Q&A uses text-only prompts either way. Models are
 * loaded from GenieX's on-device store — pulled from Qualcomm AI Hub, or
 * registered from a local bundle folder via [registerLocalModel].
 *
 * Blocking wrappers are meant to be called off the UI thread; they bridge
 * GenieX's Kotlin coroutine/Flow API with runBlocking.
 */
class GenieXRunner(private val context: Context) {
    private val tag = "GenieXRunner"

    @Volatile
    private var sdkReady = false
    private var llm: LlmWrapper? = null
    private var vlm: VlmWrapper? = null

    private suspend fun ensureInit() {
        if (sdkReady) return
        suspendCoroutine { cont ->
            GenieXSdk.getInstance().init(context, object : GenieXSdk.InitCallback {
                override fun onSuccess() = cont.resume(Unit)
                override fun onFailure(reason: String) =
                    cont.resumeWithException(RuntimeException("GenieX init failed: $reason"))
            })
        }
        val storeDir = context.getExternalFilesDir(null)?.absolutePath
            ?: context.filesDir.absolutePath
        ModelManagerWrapper.init(storeDir)
        sdkReady = true
    }

    fun isModelReady(): Boolean = runBlocking {
        ensureInit()
        ModelManagerWrapper.list().isNotEmpty()
    }

    fun listModels(): List<String> = runBlocking {
        ensureInit()
        ModelManagerWrapper.list()
    }

    /** Downloads a GenieX model (by Qualcomm AI Hub name) into the on-device store. */
    fun downloadModel(name: String): Unit = runBlocking {
        ensureInit()
        pull(name, HubSource.AIHUB, localPath = "", type = ModelType.LLM)
        resetWrappers()
    }

    /**
     * Registers a local GenieX bundle folder (e.g. /sdcard/models/<bundle>) into
     * the store via a LOCALFS pull. Requires all-files storage access to read it.
     */
    fun registerLocalModel(path: String): Unit = runBlocking {
        ensureInit()
        val dir = File(path)
        if (!File(dir, "genie_config.json").exists()) {
            throw RuntimeException("Not a GenieX bundle (no genie_config.json): $path")
        }
        val name = dir.name
        if (ModelManagerWrapper.list().contains(name)) return@runBlocking
        // Vision-language bundles ship a vision encoder; plain LLMs do not.
        val type = if (File(dir, "vision_encoder.bin").exists()) ModelType.VLM else ModelType.LLM
        pull(name, HubSource.LOCALFS, localPath = path, type = type)
        resetWrappers()
    }

    private suspend fun pull(name: String, hub: HubSource, localPath: String, type: ModelType) {
        val chipset = ModelManagerWrapper.detectChipset()
        val input = ModelPullInput(name, "", hub, localPath, "", chipset, name, type)
        ModelManagerWrapper.pullFlow(input).collect { event ->
            if (event is ModelManagerWrapper.PullEvent.Error) {
                throw RuntimeException("GenieX pull failed for $name")
            }
        }
    }

    fun generate(prompt: String): String = runBlocking {
        ensureInit()
        val models = ModelManagerWrapper.list()
        if (models.isEmpty()) {
            throw ModelUnavailableException("No GenieX model available on this device")
        }
        val name = models.first()
        when (ModelManagerWrapper.getType(name)) {
            ModelType.VLM -> generateVlm(name, prompt)
            else -> generateLlm(name, prompt)
        }
    }

    private suspend fun generateLlm(name: String, prompt: String): String {
        val w = llm ?: buildLlm(name).also { llm = it }
        val messages = arrayOf(ChatMessage("user", prompt))
        val templated = w.applyChatTemplate(messages, null, false).getOrThrow()
        return collectStream(w.generateStreamFlow(templated.formattedText, generationConfig()))
    }

    private suspend fun generateVlm(name: String, prompt: String): String {
        val w = vlm ?: buildVlm(name).also { vlm = it }
        // Text-only message (no image) — Qwen2.5-VL answers text prompts fine.
        val messages = arrayOf(VlmChatMessage("user", listOf(VlmContent("text", prompt))))
        val templated = w.applyChatTemplate(messages, null, false).getOrThrow()
        val cfg = w.injectMediaPathsToConfig(messages, generationConfig())
        return collectStream(w.generateStreamFlow(templated.formattedText, cfg))
    }

    private suspend fun collectStream(
        flow: kotlinx.coroutines.flow.Flow<LlmStreamResult>,
    ): String {
        val out = StringBuilder()
        flow.collect { r ->
            when (r) {
                is LlmStreamResult.Token -> out.append(r.text)
                is LlmStreamResult.Error -> throw r.throwable
                else -> {}
            }
        }
        return out.toString().trim()
    }

    private suspend fun buildLlm(name: String): LlmWrapper {
        val paths = ModelManagerWrapper.getPaths(name)
            ?: throw ModelUnavailableException("Model paths unavailable for $name")
        val input = LlmCreateInput(
            model_name = paths.model_name,
            model_path = paths.model_path,
            tokenizer_path = paths.tokenizer_path,
            config = ModelConfig(nCtx = 0, nGpuLayers = 0, enable_thinking = false),
            runtime_id = paths.runtime_id,
            compute_unit = ComputeUnitValue.NPU.value,
        )
        Log.i(tag, "Loading GenieX LLM ${paths.model_name} on NPU")
        return LlmWrapper.builder().llmCreateInput(input).build().getOrThrow()
    }

    private suspend fun buildVlm(name: String): VlmWrapper {
        val paths = ModelManagerWrapper.getPaths(name)
            ?: throw ModelUnavailableException("Model paths unavailable for $name")
        val input = VlmCreateInput(
            paths.model_name,
            paths.model_path,
            paths.mmproj_path, // vision encoder
            ModelConfig(nCtx = 0, nGpuLayers = 0, enable_thinking = false),
            paths.runtime_id,
            ComputeUnitValue.NPU.value,
        )
        Log.i(tag, "Loading GenieX VLM ${paths.model_name} on NPU")
        return VlmWrapper.builder().vlmCreateInput(input).build().getOrThrow()
    }

    private fun generationConfig(): GenerationConfig {
        val sampler = SamplerConfig(0.7f, 0.95f, 40, 0.0f, 1.1f, 0.0f, 0.0f, 0, "", "")
        return GenerationConfig(512, arrayOf(), 0, 0, sampler, arrayOf(), 0, arrayOf(), 0)
    }

    private fun resetWrappers() {
        llm?.close(); llm = null
        vlm?.close(); vlm = null
    }

    fun close() = resetWrappers()
}
