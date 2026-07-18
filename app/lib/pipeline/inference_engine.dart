/// Runs on-device model inference to answer questions, grounded in stored
/// memories. Swappable like every other pipeline stage — the engine decides
/// which model runs and on what hardware (CPU / NPU / remote).
abstract interface class InferenceEngine {
  /// Answers [question] and returns the model output.
  Future<String> ask(String question);

  /// Whether a generative model is loaded/available on the device.
  Future<bool> isModelReady();

  /// Downloads/prepares a model by name (e.g. a Qualcomm AI Hub model id).
  Future<void> downloadModel(String modelName);

  /// Registers a model from a local bundle folder on the device.
  Future<void> registerLocalModel(String path);

  /// Whether the engine can read local model bundles (all-files storage access).
  Future<bool> hasStorageAccess();

  /// Prompts the user to grant all-files storage access.
  Future<void> requestStorageAccess();

  Future<void> dispose();
}
