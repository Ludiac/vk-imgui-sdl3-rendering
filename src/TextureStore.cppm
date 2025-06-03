module;

#include "macros.hpp"
#include "primitive_types.hpp"

export module vulkan_app:TextureStore;

import vulkan_hpp;
import std;
import :VulkanDevice;
import :texture;
import :ModelLoader; // For GltfImageData

export class TextureStore {
private:
  VulkanDevice &device_;
  vk::raii::CommandPool ownedCommandPool_{nullptr};
  const vk::raii::Queue &transferQueue_;

  std::map<std::string, std::shared_ptr<Texture>> loadedTextures_;
  std::shared_ptr<Texture> defaultWhiteTexture_;

public:
  [[nodiscard]] std::expected<void, std::string> createInternalCommandPool() { /* ... same ... */
    if (!*device_.logical() || device_.queueFamily_ == static_cast<u32>(-1))
      return std::unexpected("TS:createPool: Device/QF not init.");
    vk::CommandPoolCreateInfo poolInfo{.flags = vk::CommandPoolCreateFlagBits::eTransient |
                                                vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                       .queueFamilyIndex = device_.queueFamily_};
    auto poolResult = device_.logical().createCommandPool(poolInfo);
    if (!poolResult)
      return std::unexpected("TS:createPool: Pool creation failed: " +
                             vk::to_string(poolResult.error()));
    ownedCommandPool_ = std::move(poolResult.value());

    auto defaultTexResult = createDefaultTexture(device_, ownedCommandPool_, transferQueue_,
                                                 vk::Format::eR8G8B8A8Unorm, {255, 255, 255, 255});
    if (defaultTexResult)
      defaultWhiteTexture_ = std::make_shared<Texture>(std::move(*defaultTexResult));
    else
      std::println("CRITICAL: TS failed to create magenta default: {}", defaultTexResult.error());

    return {};
  }

  TextureStore(VulkanDevice &device, const vk::raii::Queue &queue)
      : device_(device), transferQueue_(queue) {}

  // Load texture from raw pixel data
  [[nodiscard]] std::shared_ptr<Texture>
  getTextureFromData(const std::string &cacheKey,    // Unique key for this texture data
                     const GltfImageData &imageData) // Pass the struct from ModelLoader
  {
    if (!*ownedCommandPool_) {
      std::println("TS::getTextureFromData: No command pool. Returning fallback default.");
      return defaultWhiteTexture_;
    }
    if (imageData.pixels.empty() || imageData.width == 0 || imageData.height == 0) {
      std::println(
          "TS::getTextureFromData: Invalid image data for key '{}'. Returning fallback default.",
          cacheKey);
      return defaultWhiteTexture_;
    }

    if (auto it = loadedTextures_.find(cacheKey); it != loadedTextures_.end()) {
      return it->second;
    }

    vk::Format format = vk::Format::eR8G8B8A8Unorm; // Default
    if (imageData.component == 4) {
      format = imageData.isSrgb ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm;
    } else if (imageData.component == 3) {
      // Vulkan often doesn't have great direct support for R8G8B8.
      // It's common to convert to RGBA8 by adding an alpha channel.
      // For now, let's assume if component is 3, we might need to handle it or expect RGBA.
      // Or, if your createTexture can handle it (e.g. by reformatting during staging upload).
      // Let's assume for now imageData.pixels is already RGBA if component is 3 for simplicity,
      // or that createTexture will handle it.
      // This is a common pain point. For now, assume RGBA8 if component is not 4.
      std::println("Warning: Texture '{}' has {} components. Assuming RGBA for format selection.",
                   cacheKey, imageData.component);
      format = imageData.isSrgb ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm;
    } else {
      std::println(
          "Warning: Texture '{}' has unsupported component count {}. Using default format.",
          cacheKey, imageData.component);
    }

    auto texResult = createTexture3( // This is your existing createTexture from vulkan_app:texture
        device_, imageData.pixels.data(), imageData.pixels.size(),
        vk::Extent3D{static_cast<uint32_t>(imageData.width),
                     static_cast<uint32_t>(imageData.height), 1},
        format,
        ownedCommandPool_, // Use owned pool
        transferQueue_,
        true // generateMipmaps
    );

    if (texResult) {
      std::shared_ptr<Texture> newTexture = std::make_shared<Texture>(std::move(*texResult));
      loadedTextures_[cacheKey] = newTexture;
      std::println("TextureStore: Successfully created texture '{}' from data.", cacheKey);
      return newTexture;
    } else {
      std::println(
          "TextureStore: Failed to create texture '{}' from data: {}. Returning fallback default.",
          cacheKey, texResult.error());
      return defaultWhiteTexture_;
    }
  }

  [[nodiscard]] std::shared_ptr<Texture>
  getColorTexture(const std::string &key, std::array<uint8_t, 4> color,
                  vk::Format format = vk::Format::eR8G8B8A8Srgb) { /* ... same ... */
    if (!*ownedCommandPool_)
      return defaultWhiteTexture_;
    if (auto it = loadedTextures_.find(key); it != loadedTextures_.end())
      return it->second;
    auto texResult =
        createDefaultTexture(device_, ownedCommandPool_, transferQueue_, format, color);
    if (texResult) {
      auto newTex = std::make_shared<Texture>(std::move(*texResult));
      loadedTextures_[key] = newTex;
      return newTex;
    }
    std::println("TS: Failed to create color texture '{}': {}. Returning fallback.", key,
                 texResult.error());
    return defaultWhiteTexture_;
  }

  std::shared_ptr<Texture> getDefaultTexture() const { return defaultWhiteTexture_; }

  ~TextureStore() = default; // RAII for ownedCommandPool_ and shared_ptrs
  TextureStore(const TextureStore &) = delete;
  TextureStore &operator=(const TextureStore &) = delete;
  TextureStore(TextureStore &&) = delete;
  TextureStore &operator=(TextureStore &&) = delete;
};
