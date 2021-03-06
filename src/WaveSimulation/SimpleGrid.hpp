#pragma once

#include <stddef.h>
#include <vector>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include "Core/VkMesh.hpp"

namespace WaveSimulation {
	// Accessed with SimpleGrid[y][x]
	struct SimpleGrid {
		void init( const char* start_condition = nullptr );
		size_t get_buffer_float_amount();
		void fill_buffer( float* buffer, bool drawU );
		
		void update_ghosts( void(*)( glm::vec3& cell, size_t x, size_t y ));

		//Central difference d2x d2y forward difference d2t
		void step_finite_difference( double dt );
		void step_finite_volume( double dt );

		//FV / DG
		glm::vec3 solveRiemann( size_t xl, size_t yl, size_t xr, size_t yr, double dT, glm::vec2 normal );

		size_t x_s{ 2 };
		size_t y_s{ 2 };

		double K0{ 0.25 };
		double onebyrho0{ 1 };

		//std::vector<double> oval;   //t - dt
		std::vector<glm::vec3> values; //t
		std::vector<glm::vec3> nval;   //t + dt

		inline glm::vec3* operator[]( size_t y ){
			return &values[ y * x_s ];
		}

		static VertexInputDescription get_vk_description();
	};

	struct Riemann2Cell {
		glm::vec4 p;
		glm::vec4 ux;
		glm::vec4 uy;
	};

	struct Riemann2Grid {
		void init(const char* start_condition = nullptr);
		size_t get_buffer_float_amount();
		void fill_buffer(float* buffer, bool drawU);

		void update_ghosts(void(*)(Riemann2Cell& cell, size_t x, size_t y));

		//Central difference d2x d2y forward difference d2t
		void step_finite_difference(double dt);
		void step_finite_volume(double dt);

		//FV / DG
		glm::vec3 solveRiemann(glm::vec3 left, glm::vec3 right, double dT, glm::vec2 normal);

		size_t x_s{ 2 };
		size_t y_s{ 2 };

		double K0{ 1 };
		double onebyrho0{ 1 };

		std::vector<Riemann2Cell> values; //t
		std::vector<Riemann2Cell> nval;   //t + dt

		inline Riemann2Cell* operator[](size_t y) {
			return &values[y * x_s];
		}

		static VertexInputDescription get_vk_description();
	};
}
