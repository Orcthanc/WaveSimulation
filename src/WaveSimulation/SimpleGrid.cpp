#include "SimpleGrid.hpp"

#include "stb_image.h"
#include <glm/geometric.hpp>
#include <iostream>
#include <string.h>

#include <glm/vec3.hpp>
#include <vulkan/vulkan_core.h>

using namespace WaveSimulation;

void SimpleGrid::init( const char* start_condition ){
	int width, height, channels;

	stbi_uc* data = stbi_load( start_condition, &width, &height, &channels, STBI_grey );

	if( !data ){
		std::cout << "Failed to load texture " << start_condition << " because of: " << stbi_failure_reason() << std::endl;
		return;
	}

	x_s = width;
	y_s = height;

	values.resize( x_s * y_s );

	for( size_t i = 0; i < x_s * y_s; ++i )
		values[i] = data[i] / 255.0;

	stbi_image_free( data );
}

size_t SimpleGrid::get_buffer_float_amount(){
	return (x_s - 1) * (y_s - 1) * 36;
}

void SimpleGrid::fill_buffer( float* buffer ){
	constexpr float yscale = 0.1;
	const float xscale = 2.0 / x_s;
	const float zscale = 2.0 / y_s;

	size_t index = 0;

	for( size_t y = 0; y < y_s - 1; ++y ){
		for( size_t x = 0; x < x_s - 1; ++x ){
			float dx1 = (*this)[y][x + 1] - (*this)[y][x];
			float dy1 = (*this)[y + 1][x] - (*this)[y][x];

			float dx2 = (*this)[y + 1][x + 1] - (*this)[y + 1][x];
			float dy2 = (*this)[y + 1][x + 1] - (*this)[y][x + 1];

			glm::vec3 norm1 = glm::normalize( glm::vec3{ dx1, 1, dy1 });
			glm::vec3 norm2 = glm::normalize( glm::vec3{ dx2, 1, dy2 });

			// Vert 1
			buffer[index++] = x * xscale - 1; //x
			buffer[index++] = (*this)[y][x] * yscale; //y
			buffer[index++] = y * zscale - 1; //z

			buffer[index++] = norm1.x; //x-normal
			buffer[index++] = norm1.y; //y-normal
			buffer[index++] = norm1.z; //z-normal

			// Vert 2
			buffer[index++] = x * xscale - 1 + xscale; //x
			buffer[index++] = (*this)[y][x + 1] * yscale; //y
			buffer[index++] = y * zscale - 1; //z

			buffer[index++] = norm1.x; //x-normal
			buffer[index++] = norm1.y; //y-normal
			buffer[index++] = norm1.z; //z-normal

			// Vert 3
			buffer[index++] = x * xscale - 1; //x
			buffer[index++] = (*this)[y + 1][x] * yscale; //y
			buffer[index++] = y * zscale - 1 + zscale; //z

			buffer[index++] = norm1.x; //x-normal
			buffer[index++] = norm1.y; //y-normal
			buffer[index++] = norm1.z; //z-normal

			// Vert 4
			buffer[index++] = x * xscale - 1 + xscale; //x
			buffer[index++] = (*this)[y][x + 1] * yscale; //y
			buffer[index++] = y * zscale - 1; //z

			buffer[index++] = norm2.x; //x-normal
			buffer[index++] = norm2.y; //y-normal
			buffer[index++] = norm2.z; //z-normal

			// Vert 5
			buffer[index++] = x * xscale - 1; //x
			buffer[index++] = (*this)[y + 1][x] * yscale; //y
			buffer[index++] = y * zscale - 1 + zscale; //z

			buffer[index++] = norm2.x; //x-normal
			buffer[index++] = norm2.y; //y-normal
			buffer[index++] = norm2.z; //z-normal

			// Vert 6
			buffer[index++] = x * xscale - 1 + xscale; //x
			buffer[index++] = (*this)[y + 1][x + 1] * yscale; //y
			buffer[index++] = y * zscale - 1 + zscale; //z

			buffer[index++] = norm2.x; //x-normal
			buffer[index++] = norm2.y; //y-normal
			buffer[index++] = norm2.z; //z-normal

		}
	}
}

VertexInputDescription SimpleGrid::get_vk_description(){
	VertexInputDescription desc;

	desc.bindings.emplace_back( VkVertexInputBindingDescription{
			.binding = 0,
			.stride = 6 * sizeof( float ),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		});

	desc.attributes.emplace_back( VkVertexInputAttributeDescription{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = 0,
		});

	desc.attributes.emplace_back( VkVertexInputAttributeDescription{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = 3 * sizeof( float ),
		});

	return desc;
}
