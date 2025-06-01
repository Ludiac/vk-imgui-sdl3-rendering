module;

#include "macros.hpp"
#include "primitive_types.hpp"

export module vulkan_app:VMA;

import vulkan_hpp;
import std;
import vk_mem_alloc_hpp;

// RAII Wrapper for a VMA-allocated buffer
export struct VmaBuffer {
  vma::Allocator allocator_{nullptr}; // Store the allocator that created this buffer
  vk::Buffer buffer_{nullptr};
  vma::Allocation allocation_{nullptr};
  vma::AllocationInfo allocationInfo_{};
  vk::DeviceSize size_ = 0;     // Store size for potential re-creation or info
  void *pMappedData_ = nullptr; // If persistently mapped

  // Default constructor: creates an empty, invalid buffer
  VmaBuffer() = default;

  // Constructor to be used by factory functions (like VulkanDevice::createBufferWithVma)
  VmaBuffer(vma::Allocator allocator, vk::Buffer buffer, vma::Allocation allocation,
            const vma::AllocationInfo &allocInfo, vk::DeviceSize size)
      : allocator_(allocator), buffer_(buffer), allocation_(allocation), allocationInfo_(allocInfo),
        size_(size) {
    // If the allocation was created with VMA_ALLOCATION_CREATE_MAPPED_BIT, pMappedData is already
    // valid.
    if (allocInfo.pMappedData != nullptr) {
      pMappedData_ = allocInfo.pMappedData;
    }
  }

  // Destructor: automatically cleans up the buffer and allocation
  ~VmaBuffer() { release(); }

  // Move constructor
  VmaBuffer(VmaBuffer &&other) noexcept
      : allocator_(other.allocator_), buffer_(other.buffer_), allocation_(other.allocation_),
        allocationInfo_(other.allocationInfo_), size_(other.size_),
        pMappedData_(other.pMappedData_) {
    // Reset other to prevent double deletion
    other.allocator_ = nullptr;
    other.buffer_ = nullptr;
    other.allocation_ = nullptr;
    other.pMappedData_ = nullptr;
    other.size_ = 0;
  }

  // Move assignment operator
  VmaBuffer &operator=(VmaBuffer &&other) noexcept {
    if (this != &other) {
      release(); // Clean up existing resource

      allocator_ = other.allocator_;
      buffer_ = other.buffer_;
      allocation_ = other.allocation_;
      allocationInfo_ = other.allocationInfo_;
      size_ = other.size_;
      pMappedData_ = other.pMappedData_;

      other.allocator_ = nullptr;
      other.buffer_ = nullptr;
      other.allocation_ = nullptr;
      other.pMappedData_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  // Delete copy constructor and copy assignment operator
  VmaBuffer(const VmaBuffer &) = delete;
  VmaBuffer &operator=(const VmaBuffer &) = delete;

  // Explicitly release resources
  void release() {
    if (allocator_ && buffer_ && allocation_) {
      // If memory was mapped by us (not persistently by VMA), unmap it first.
      // However, if pMappedData_ was set because VMA_ALLOCATION_CREATE_MAPPED_BIT was used,
      // vmaDestroyBuffer handles it. If we called vmaMapMemory, we should call vmaUnmapMemory.
      // For simplicity, this RAII wrapper assumes if pMappedData_ is set via
      // VMA_ALLOCATION_CREATE_MAPPED_BIT, vmaDestroyBuffer is sufficient. If map/unmap is manual,
      // the user of VmaBuffer should handle it or this class needs map/unmap methods. If
      // pMappedData_ is set and it wasn't from VMA_ALLOCATION_CREATE_MAPPED_BIT, it implies manual
      // mapping. This RAII wrapper doesn't manage that state perfectly without more info. For now,
      // assume vmaDestroyBuffer is sufficient.

      allocator_.destroyBuffer(buffer_, allocation_);
    }
    allocator_ = nullptr;
    buffer_ = nullptr;
    allocation_ = nullptr;
    pMappedData_ = nullptr;
    size_ = 0;
    allocationInfo_ = vma::AllocationInfo{};
  }

  // Accessors
  vk::Buffer get() const { return buffer_; }
  vma::Allocation getAllocation() const { return allocation_; }
  const vma::AllocationInfo &getAllocationInfo() const { return allocationInfo_; }
  vk::DeviceSize getSize() const { return size_; }
  void *getMappedData() const {
    return pMappedData_;
  } // Returns persistently mapped pointer, if available

  explicit operator bool() const { return buffer_ && allocation_; }

  // operator vk::Buffer() const { return vk::Buffer(buffer_); } // If you want implicit conversion
  // to vk::Buffer
};

// RAII Wrapper for a VMA-allocated image
export struct VmaImage {
  vma::Allocator allocator_ = nullptr;
  vk::Image image_ = nullptr;
  vma::Allocation allocation_ = nullptr;
  vma::AllocationInfo allocationInfo_{};
  vk::Format format_ = vk::Format::eUndefined;
  vk::Extent3D extent_ = {0, 0, 0};

  VmaImage() = default;

  VmaImage(vma::Allocator allocator, vk::Image image, vma::Allocation allocation,
           const vma::AllocationInfo &allocInfo, vk::Format format, vk::Extent3D extent)
      : allocator_(allocator), image_(image), allocation_(allocation), allocationInfo_(allocInfo),
        format_(format), extent_(extent) {}

  ~VmaImage() { release(); }

  VmaImage(VmaImage &&other) noexcept
      : allocator_(other.allocator_), image_(other.image_), allocation_(other.allocation_),
        allocationInfo_(other.allocationInfo_), format_(other.format_), extent_(other.extent_) {
    other.allocator_ = nullptr;
    other.image_ = nullptr;
    other.allocation_ = nullptr;
    other.format_ = vk::Format::eUndefined;
    other.extent_ = {0, 0, 0};
  }

  VmaImage &operator=(VmaImage &&other) noexcept {
    if (this != &other) {
      release();
      allocator_ = other.allocator_;
      image_ = other.image_;
      allocation_ = other.allocation_;
      allocationInfo_ = other.allocationInfo_;
      format_ = other.format_;
      extent_ = other.extent_;

      other.allocator_ = nullptr;
      other.image_ = nullptr;
      other.allocation_ = nullptr;
      other.format_ = vk::Format::eUndefined;
      other.extent_ = {0, 0, 0};
    }
    return *this;
  }

  VmaImage(const VmaImage &) = delete;
  VmaImage &operator=(const VmaImage &) = delete;

  void release() {
    if (allocator_ && image_ && allocation_) {
      allocator_.destroyImage(image_, allocation_);
    }
    allocator_ = nullptr;
    image_ = nullptr;
    allocation_ = nullptr;
    format_ = vk::Format::eUndefined;
    extent_ = {0, 0, 0};
    allocationInfo_ = vma::AllocationInfo{};
  }

  vk::Image get() const { return image_; }
  vma::Allocation getAllocation() const { return allocation_; }
  const vma::AllocationInfo &getAllocationInfo() const { return allocationInfo_; }
  vk::Format getFormat() const { return format_; }
  vk::Extent3D getExtent() const { return extent_; }

  explicit operator bool() const { return image_ && allocation_; }
  // operator vk::Image() const { return vk::Image(image_); }
};

// Helper function to initialize the VMA Allocator (remains the same)
export std::expected<vma::Allocator, std::string>
createVmaAllocator(vk::Instance instance, vk::PhysicalDevice physicalDevice, vk::Device device) {
  uint32_t apiVersion = vk::makeApiVersion(0, 1, 0, 0);
  auto instanceVersionResult = vk::enumerateInstanceVersion();
  if (instanceVersionResult.result == vk::Result::eSuccess) {
    apiVersion = instanceVersionResult.value;
  }
  vma::AllocatorCreateInfo allocatorInfo = {
      .physicalDevice = static_cast<vk::PhysicalDevice>(physicalDevice),
      .device = static_cast<vk::Device>(device),
      .instance = static_cast<vk::Instance>(instance),
      .vulkanApiVersion = apiVersion,
  };

  vma::Allocator allocator_handle; // Renamed to avoid conflict with member
  vk::Result result = vma::createAllocator(&allocatorInfo, &allocator_handle);
  if (result != vk::Result::eSuccess) {
    return std::unexpected("Failed to create VMA allocator: " +
                           vk::to_string(static_cast<vk::Result>(result)));
  }
  return allocator_handle;
}
