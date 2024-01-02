#if defined(_WIN32)
#pragma comment(linker, "/subsystem:console")

#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <fstream>

#include <vulkan/vulkan.h>

std::string errorString(VkResult errorCode)
{
    switch (errorCode)
    {
#define STR(r) case VK_ ##r: return #r
        STR(NOT_READY);
        STR(TIMEOUT);
        STR(EVENT_SET);
        STR(EVENT_RESET);
        STR(INCOMPLETE);
        STR(ERROR_OUT_OF_HOST_MEMORY);
        STR(ERROR_OUT_OF_DEVICE_MEMORY);
        STR(ERROR_INITIALIZATION_FAILED);
        STR(ERROR_DEVICE_LOST);
        STR(ERROR_MEMORY_MAP_FAILED);
        STR(ERROR_LAYER_NOT_PRESENT);
        STR(ERROR_EXTENSION_NOT_PRESENT);
        STR(ERROR_FEATURE_NOT_PRESENT);
        STR(ERROR_INCOMPATIBLE_DRIVER);
        STR(ERROR_TOO_MANY_OBJECTS);
        STR(ERROR_FORMAT_NOT_SUPPORTED);
        STR(ERROR_SURFACE_LOST_KHR);
        STR(ERROR_NATIVE_WINDOW_IN_USE_KHR);
        STR(SUBOPTIMAL_KHR);
        STR(ERROR_OUT_OF_DATE_KHR);
        STR(ERROR_INCOMPATIBLE_DISPLAY_KHR);
        STR(ERROR_VALIDATION_FAILED_EXT);
        STR(ERROR_INVALID_SHADER_NV);
        STR(ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
#undef STR
    default:
        return "UNKNOWN_ERROR";
    }
}


#define VK_CHECK_RESULT(f)																				\
{																										\
	VkResult res = (f);																					\
	if (res != VK_SUCCESS)																				\
	{																									\
		std::cout << "Fatal : VkResult is \"" << errorString(res) << "\" in " << __FILE__ << " at line " << __LINE__ << "\n"; \
		assert(res == VK_SUCCESS);																		\
	}																									\
}

VkShaderModule loadShader(const char *fileName, VkDevice device)
{
    std::ifstream is(fileName, std::ios::binary | std::ios::in | std::ios::ate);

    if (is.is_open())
    {
        size_t size = is.tellg();
        is.seekg(0, std::ios::beg);
        char* shaderCode = new char[size];
        is.read(shaderCode, size);
        is.close();

        assert(size > 0);

        VkShaderModule shaderModule;
        VkShaderModuleCreateInfo moduleCreateInfo{};
        moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        moduleCreateInfo.codeSize = size;
        moduleCreateInfo.pCode = (uint32_t*)shaderCode;

        VK_CHECK_RESULT(vkCreateShaderModule(device, &moduleCreateInfo, NULL, &shaderModule));

        delete[] shaderCode;

        return shaderModule;
    }
    else
    {
        std::cerr << "Error: Could not open shader file \"" << fileName << "\"" << "\n";
        return VK_NULL_HANDLE;
    }
}
#define BUFFER_ELEMENTS 32

#define LOG(...) printf(__VA_ARGS__)

static VKAPI_ATTR VkBool32 VKAPI_CALL debugMessageCallback(
	VkDebugReportFlagsEXT flags,
	VkDebugReportObjectTypeEXT objectType,
	uint64_t object,
	size_t location,
	int32_t messageCode,
	const char* pLayerPrefix,
	const char* pMessage,
	void* pUserData)
{
	LOG("[VALIDATION]: %s - %s\n", pLayerPrefix, pMessage);
	return VK_FALSE;
}


class VulkanExample
{
public:
	VkInstance instance;
	VkPhysicalDevice physicalDevice;
	VkDevice device;
	uint32_t queueFamilyIndex;
	VkPipelineCache pipelineCache;
	VkQueue queue;
	VkCommandPool commandPool;
	VkCommandBuffer commandBuffer;
	VkFence fence;
	VkDescriptorPool descriptorPool;
	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorSet descriptorSet;
	VkPipelineLayout pipelineLayout;
	VkPipeline pipeline;
	VkShaderModule shaderModule;

	VkDebugReportCallbackEXT debugReportCallback{};

	VkResult createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkBuffer *buffer, VkDeviceMemory *memory, VkDeviceSize size, void *data = nullptr)
	{
		// Create the buffer handle
        VkBufferCreateInfo bufferCreateInfo {};
        bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.usage = usageFlags;
        bufferCreateInfo.size = size;

		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, nullptr, buffer));

		// Create the memory backing up the buffer handle
		VkPhysicalDeviceMemoryProperties deviceMemoryProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &deviceMemoryProperties);
		VkMemoryRequirements memReqs;
    VkMemoryAllocateInfo memAlloc {};
        memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        vkGetBufferMemoryRequirements(device, *buffer, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		// Find a memory type index that fits the properties of the buffer
		bool memTypeFound = false;
		for (uint32_t i = 0; i < deviceMemoryProperties.memoryTypeCount; i++) {
			if ((memReqs.memoryTypeBits & 1) == 1) {
				if ((deviceMemoryProperties.memoryTypes[i].propertyFlags & memoryPropertyFlags) == memoryPropertyFlags) {
					memAlloc.memoryTypeIndex = i;
					memTypeFound = true;
					break;
				}
			}
			memReqs.memoryTypeBits >>= 1;
		}
		assert(memTypeFound);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, memory));

		if (data != nullptr) {
			void *mapped;
			VK_CHECK_RESULT(vkMapMemory(device, *memory, 0, size, 0, &mapped));
			memcpy(mapped, data, size);
			vkUnmapMemory(device, *memory);
		}

		VK_CHECK_RESULT(vkBindBufferMemory(device, *buffer, *memory, 0));

		return VK_SUCCESS;
	}

	VulkanExample()
	{
		LOG("Running headless compute example\n");

		VkApplicationInfo appInfo = {};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Vulkan headless example";
		appInfo.pEngineName = "VulkanExample";
		appInfo.apiVersion = VK_API_VERSION_1_0;

		/*
			Vulkan instance creation (without surface extensions)
		*/
		VkInstanceCreateInfo instanceCreateInfo = {};
		instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		instanceCreateInfo.pApplicationInfo = &appInfo;

		uint32_t layerCount = 1;
		const char* validationLayers[] = { "VK_LAYER_KHRONOS_validation" };

		std::vector<const char*> instanceExtensions = {};

		// Check if layers are available
		uint32_t instanceLayerCount;
		vkEnumerateInstanceLayerProperties(&instanceLayerCount, nullptr);
		std::vector<VkLayerProperties> instanceLayers(instanceLayerCount);
		vkEnumerateInstanceLayerProperties(&instanceLayerCount, instanceLayers.data());

		bool layersAvailable = true;
		for (auto layerName : validationLayers) {
			bool layerAvailable = false;
			for (auto instanceLayer : instanceLayers) {
				if (strcmp(instanceLayer.layerName, layerName) == 0) {
					layerAvailable = true;
					break;
				}
			}
			if (!layerAvailable) {
				layersAvailable = false;
				break;
			}
		}

		if (layersAvailable) {
			instanceExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
			instanceCreateInfo.ppEnabledLayerNames = validationLayers;
			instanceCreateInfo.enabledLayerCount = layerCount;
		}

		instanceCreateInfo.enabledExtensionCount = (uint32_t)instanceExtensions.size();
		instanceCreateInfo.ppEnabledExtensionNames = instanceExtensions.data();
		VK_CHECK_RESULT(vkCreateInstance(&instanceCreateInfo, nullptr, &instance));


		if (layersAvailable) {
			VkDebugReportCallbackCreateInfoEXT debugReportCreateInfo = {};
			debugReportCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
			debugReportCreateInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
			debugReportCreateInfo.pfnCallback = (PFN_vkDebugReportCallbackEXT)debugMessageCallback;

			// We have to explicitly load this function.
			PFN_vkCreateDebugReportCallbackEXT vkCreateDebugReportCallbackEXT = reinterpret_cast<PFN_vkCreateDebugReportCallbackEXT>(vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT"));
			assert(vkCreateDebugReportCallbackEXT);
			VK_CHECK_RESULT(vkCreateDebugReportCallbackEXT(instance, &debugReportCreateInfo, nullptr, &debugReportCallback));
		}

		/*
			Vulkan device creation
		*/
		// Physical device (always use first)
		uint32_t deviceCount = 0;
		VK_CHECK_RESULT(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr));
		std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
		VK_CHECK_RESULT(vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data()));
		physicalDevice = physicalDevices[0];

		VkPhysicalDeviceProperties deviceProperties;
		vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
		LOG("GPU: %s\n", deviceProperties.deviceName);

		// Request a single compute queue
		const float defaultQueuePriority(0.0f);
		VkDeviceQueueCreateInfo queueCreateInfo = {};
		uint32_t queueFamilyCount;
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
		std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilyProperties.data());
		for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++) {
			if (queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
				queueFamilyIndex = i;
				queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
				queueCreateInfo.queueFamilyIndex = i;
				queueCreateInfo.queueCount = 1;
				queueCreateInfo.pQueuePriorities = &defaultQueuePriority;
				break;
			}
		}
		// Create logical device
		VkDeviceCreateInfo deviceCreateInfo = {};
		deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		deviceCreateInfo.queueCreateInfoCount = 1;
		deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
		std::vector<const char*> deviceExtensions = {};

		deviceCreateInfo.enabledExtensionCount = (uint32_t)deviceExtensions.size();
		deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
		VK_CHECK_RESULT(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device));

		// Get a compute queue
		vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

		// Compute command pool
		VkCommandPoolCreateInfo cmdPoolInfo = {};
		cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmdPoolInfo.queueFamilyIndex = queueFamilyIndex;
		cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &commandPool));

		/*
			Prepare storage buffers
		*/
		std::vector<uint32_t> computeInput(BUFFER_ELEMENTS);
		std::vector<uint32_t> computeOutput(BUFFER_ELEMENTS);

		// Fill input data
		uint32_t n = 0;
		std::generate(computeInput.begin(), computeInput.end(), [&n] { return n++; });

		const VkDeviceSize bufferSize = BUFFER_ELEMENTS * sizeof(uint32_t);

		VkBuffer deviceBuffer, hostBuffer;
		VkDeviceMemory deviceMemory, hostMemory;

		// Copy input data to VRAM using a staging buffer
		{
			createBuffer(
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
				&hostBuffer,
				&hostMemory,
				bufferSize,
				computeInput.data());

			// Flush writes to host visible buffer
			void* mapped;
			vkMapMemory(device, hostMemory, 0, VK_WHOLE_SIZE, 0, &mapped);
			VkMappedMemoryRange mappedRange{};
			mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            mappedRange.memory = hostMemory;
			mappedRange.offset = 0;
			mappedRange.size = VK_WHOLE_SIZE;
			vkFlushMappedMemoryRanges(device, 1, &mappedRange);
			vkUnmapMemory(device, hostMemory);

			createBuffer(
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				&deviceBuffer,
				&deviceMemory,
				bufferSize);

			// Copy to staging buffer
			VkCommandBufferAllocateInfo cmdBufAllocateInfo {};

			cmdBufAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			cmdBufAllocateInfo.commandPool = commandPool;
			cmdBufAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			cmdBufAllocateInfo.commandBufferCount = 1;			
            VkCommandBuffer copyCmd;
			VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &copyCmd));
			VkCommandBufferBeginInfo cmdBufInfo{};
            cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			VK_CHECK_RESULT(vkBeginCommandBuffer(copyCmd, &cmdBufInfo));

			VkBufferCopy copyRegion = {};
			copyRegion.size = bufferSize;
			vkCmdCopyBuffer(copyCmd, hostBuffer, deviceBuffer, 1, &copyRegion);
			VK_CHECK_RESULT(vkEndCommandBuffer(copyCmd));

			VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &copyCmd;
			VkFenceCreateInfo fenceInfo{};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			fenceInfo.flags = 0;			
            VkFence fence;
			VK_CHECK_RESULT(vkCreateFence(device, &fenceInfo, nullptr, &fence));

			// Submit to the queue
			VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));
			VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

			vkDestroyFence(device, fence, nullptr);
			vkFreeCommandBuffers(device, commandPool, 1, &copyCmd);
		}

		/*
			Prepare compute pipeline
		*/
		{

			VkDescriptorPoolSize descriptorPoolSize {};
			descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorPoolSize.descriptorCount = 1;
            std::vector<VkDescriptorPoolSize> poolSizes = {
                descriptorPoolSize
			};

            VkDescriptorPoolCreateInfo descriptorPoolInfo {};
			descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
			descriptorPoolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
			descriptorPoolInfo.pPoolSizes = poolSizes.data();
			descriptorPoolInfo.maxSets = 1;

			VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

			VkDescriptorSetLayoutBinding setLayoutBinding {};
			setLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			setLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			setLayoutBinding.binding = 0;
			setLayoutBinding.descriptorCount = 1;
			std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
				setLayoutBinding,
			};

            VkDescriptorSetLayoutCreateInfo descriptorLayout {};
			descriptorLayout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			descriptorLayout.pBindings = setLayoutBindings.data();
			descriptorLayout.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());;

			VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout));

			VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo {};
			pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			pipelineLayoutCreateInfo.setLayoutCount = 1;
			pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;

			VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

			VkDescriptorSetAllocateInfo allocInfo {};
			allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			allocInfo.descriptorPool = descriptorPool;
			allocInfo.pSetLayouts = &descriptorSetLayout;
			allocInfo.descriptorSetCount = 1;

			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));


			VkDescriptorBufferInfo bufferDescriptor = { deviceBuffer, 0, VK_WHOLE_SIZE };
			VkWriteDescriptorSet writeDescriptorSet {};
			writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writeDescriptorSet.dstSet = descriptorSet;
			writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			writeDescriptorSet.dstBinding = 0;
			writeDescriptorSet.pBufferInfo = &bufferDescriptor;
			writeDescriptorSet.descriptorCount = 1;

			std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets = {writeDescriptorSet};

			vkUpdateDescriptorSets(device, static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, NULL);

			VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
			pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
			VK_CHECK_RESULT(vkCreatePipelineCache(device, &pipelineCacheCreateInfo, nullptr, &pipelineCache));

			// Create pipeline
            VkComputePipelineCreateInfo computePipelineCreateInfo {};
			computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
			computePipelineCreateInfo.layout = pipelineLayout;
			computePipelineCreateInfo.flags = 0;

			// Pass SSBO size via specialization constant
			struct SpecializationData {
				uint32_t BUFFER_ELEMENT_COUNT = BUFFER_ELEMENTS;
			} specializationData;
			VkSpecializationMapEntry specializationMapEntry{};
			specializationMapEntry.constantID = 0;
			specializationMapEntry.offset = 0;
			specializationMapEntry.size = sizeof(uint32_t);

			VkSpecializationInfo specializationInfo{};
			specializationInfo.mapEntryCount = 1;
			specializationInfo.pMapEntries = &specializationMapEntry;
			specializationInfo.dataSize =  sizeof(SpecializationData);
			specializationInfo.pData = &specializationData;


			VkPipelineShaderStageCreateInfo shaderStage = {};
			shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;

			shaderStage.module = loadShader("shader.spv", device);

			shaderStage.pName = "main";
			shaderStage.pSpecializationInfo = &specializationInfo;
			shaderModule = shaderStage.module;

			assert(shaderStage.module != VK_NULL_HANDLE);
			computePipelineCreateInfo.stage = shaderStage;
			VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, nullptr, &pipeline));

			// Create a command buffer for compute operations
			VkCommandBufferAllocateInfo cmdBufAllocateInfo {};
			cmdBufAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			cmdBufAllocateInfo.commandPool = commandPool;
			cmdBufAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			cmdBufAllocateInfo.commandBufferCount = 1;

			VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &commandBuffer));

			// Fence for compute CB sync
			VkFenceCreateInfo fenceCreateInfo {};
			fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

			VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, nullptr, &fence));
		}

		/*
			Command buffer creation (for compute work submission)
		*/
		{
			VkCommandBufferBeginInfo cmdBufInfo {};
			cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

			VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &cmdBufInfo));

			// Barrier to ensure that input buffer transfer is finished before compute shader reads from it
			VkBufferMemoryBarrier bufferBarrier {};
			bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
			bufferBarrier.buffer = deviceBuffer;
			bufferBarrier.size = VK_WHOLE_SIZE;
			bufferBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
			bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

			vkCmdPipelineBarrier(
				commandBuffer,
				VK_PIPELINE_STAGE_HOST_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				0,
				0, nullptr,
				1, &bufferBarrier,
				0, nullptr);

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, 0);

			vkCmdDispatch(commandBuffer, BUFFER_ELEMENTS, 1, 1);

			// Barrier to ensure that shader writes are finished before buffer is read back from GPU
			bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			bufferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			bufferBarrier.buffer = deviceBuffer;
			bufferBarrier.size = VK_WHOLE_SIZE;
			bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

			vkCmdPipelineBarrier(
				commandBuffer,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				0,
				0, nullptr,
				1, &bufferBarrier,
				0, nullptr);

			// Read back to host visible buffer
			VkBufferCopy copyRegion = {};
			copyRegion.size = bufferSize;
			vkCmdCopyBuffer(commandBuffer, deviceBuffer, hostBuffer, 1, &copyRegion);

			// Barrier to ensure that buffer copy is finished before host reading from it
			bufferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			bufferBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
			bufferBarrier.buffer = hostBuffer;
			bufferBarrier.size = VK_WHOLE_SIZE;
			bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

			vkCmdPipelineBarrier(
				commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_HOST_BIT,
				0,
				0, nullptr,
				1, &bufferBarrier,
				0, nullptr);

			VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));

			// Submit compute work
			vkResetFences(device, 1, &fence);
			const VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkSubmitInfo computeSubmitInfo {};
			computeSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
			computeSubmitInfo.commandBufferCount = 1;
			computeSubmitInfo.pCommandBuffers = &commandBuffer;
			VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &computeSubmitInfo, fence));
			VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

			// Make device writes visible to the host
			void *mapped;
			vkMapMemory(device, hostMemory, 0, VK_WHOLE_SIZE, 0, &mapped);
            VkMappedMemoryRange mappedRange {};
			mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;

			mappedRange.memory = hostMemory;
			mappedRange.offset = 0;
			mappedRange.size = VK_WHOLE_SIZE;
			vkInvalidateMappedMemoryRanges(device, 1, &mappedRange);

			// Copy to output
			memcpy(computeOutput.data(), mapped, bufferSize);
			vkUnmapMemory(device, hostMemory);
		}

		vkQueueWaitIdle(queue);

		// Output buffer contents
		LOG("Compute input:\n");
		for (auto v : computeInput) {
			LOG("%d \t", v);
		}
		std::cout << std::endl;

		LOG("Compute output:\n");
		for (auto v : computeOutput) {
			LOG("%d \t", v);
		}
		std::cout << std::endl;

		// Clean up
		vkDestroyBuffer(device, deviceBuffer, nullptr);
		vkFreeMemory(device, deviceMemory, nullptr);
		vkDestroyBuffer(device, hostBuffer, nullptr);
		vkFreeMemory(device, hostMemory, nullptr);
	}

	~VulkanExample()
	{
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		vkDestroyPipeline(device, pipeline, nullptr);
		vkDestroyPipelineCache(device, pipelineCache, nullptr);
		vkDestroyFence(device, fence, nullptr);
		vkDestroyCommandPool(device, commandPool, nullptr);
		vkDestroyShaderModule(device, shaderModule, nullptr);
		vkDestroyDevice(device, nullptr);
		if (debugReportCallback) {
			PFN_vkDestroyDebugReportCallbackEXT vkDestroyDebugReportCallback = reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT"));
			assert(vkDestroyDebugReportCallback);
			vkDestroyDebugReportCallback(instance, debugReportCallback, nullptr);
		}
		vkDestroyInstance(instance, nullptr);
	}
};


int main(int argc, char* argv[]) {
	VulkanExample *vulkanExample = new VulkanExample();
	std::cout << "Finished. Press enter to terminate...";
	std::cin.get();
	delete(vulkanExample);
	return 0;
}
