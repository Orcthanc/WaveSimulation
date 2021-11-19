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
		void set_wave_speed( double c );
		
		void update_ghosts( void(*)( double& cell, size_t x, size_t y ));

		//Central difference d2x d2y forward difference d2t
		void step_finite_difference( double dt );

		size_t x_s{ 2 };
		size_t y_s{ 2 };

		double c{ 10 };

		std::vector<double> oval;   //t - dt
		std::vector<double> values; //t
		std::vector<double> nval;   //t + dt

		inline double* operator[]( size_t y ){
			return &values[ y * x_s ];
		}

		static VertexInputDescription get_vk_description();
	};
}
