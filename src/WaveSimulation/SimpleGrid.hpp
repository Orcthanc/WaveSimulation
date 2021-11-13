#pragma once

#include <stddef.h>
#include <vector>
#include "Core/VkMesh.hpp"

namespace WaveSimulation {
	// Accessed with SimpleGrid[y][x]
	struct SimpleGrid {
		void init( const char* start_condition = nullptr );
		size_t get_buffer_float_amount();
		void fill_buffer( float* buffer );

		size_t x_s{ 2 };
		size_t y_s{ 2 };
		std::vector<double> values;

		inline double* operator[]( size_t y ){
			return &values[ y * x_s ];
		}

		static VertexInputDescription get_vk_description();
	};
}
