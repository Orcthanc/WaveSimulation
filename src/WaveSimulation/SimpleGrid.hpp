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
		
		void update_ghosts( void(*)( std::pair<double, double>& cell, size_t x, size_t y ));

		//Central difference d2x d2y forward difference d2t
		void step_finite_difference( double dt );
		void step_finite_volume( double dt );

		//FV / DG
		std::pair<double, double> solveRiemann( size_t xl, size_t yl, size_t xr, size_t yr, double dT );

		size_t x_s{ 2 };
		size_t y_s{ 2 };

		double K0{ 10 };
		double onebyrho0{ 1 };

		//std::vector<double> oval;   //t - dt
		std::vector<std::pair<double, double>> values; //t
		std::vector<std::pair<double, double>> nval;   //t + dt

		inline std::pair<double, double>* operator[]( size_t y ){
			return &values[ y * x_s ];
		}

		static VertexInputDescription get_vk_description();
	};
}
