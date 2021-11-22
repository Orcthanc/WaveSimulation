#include "SimpleGrid.hpp"

#include "stb_image.h"
#include <glm/geometric.hpp>
#include <iostream>
#include <string.h>

#include <glm/vec3.hpp>
#include <utility>
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
		values[i] = std::make_pair<double, double>( data[i] / 255.0, 0 );

	//oval = values; //Copy
	nval.resize( values.size() );

	stbi_image_free( data );
}

size_t SimpleGrid::get_buffer_float_amount(){
	return (x_s - 1) * (y_s - 1) * 36;
}

void SimpleGrid::fill_buffer( float* buffer, bool drawU ){
	constexpr float yscale = 0.3;
	const float xscale = 2.0 / x_s;
	const float zscale = 2.0 / y_s;

	size_t index = 0;

	for( size_t y = 0; y < y_s - 1; ++y ){
		for( size_t x = 0; x < x_s - 1; ++x ){
			float dx1, dy1, dx2, dy2;
			if (drawU) {
				dx1 = (*this)[y][x + 1].second - (*this)[y][x].second;
				dy1 = (*this)[y + 1][x].second - (*this)[y][x].second;

				dx2 = (*this)[y + 1][x + 1].second - (*this)[y + 1][x].second;
				dy2 = (*this)[y + 1][x + 1].second - (*this)[y][x + 1].second;
			}else {
				dx1 = (*this)[y][x + 1].first - (*this)[y][x].first;
				dy1 = (*this)[y + 1][x].first - (*this)[y][x].first;

				dx2 = (*this)[y + 1][x + 1].first - (*this)[y + 1][x].first;
				dy2 = (*this)[y + 1][x + 1].first - (*this)[y][x + 1].first;
			}

			glm::vec3 norm1 = -glm::normalize( glm::vec3{ dx1 / yscale, -1, dy1 / yscale });
			glm::vec3 norm2 = -glm::normalize( glm::vec3{ dx2 / yscale, -1, dy2 / yscale });

			// Vert 1
			buffer[index++] = x * xscale - 1; //x
			buffer[index++] = (drawU ? (*this)[y][x].second : (*this)[y][x].first) * yscale; //y
			buffer[index++] = y * zscale - 1; //z

			buffer[index++] = norm1.x; //x-normal
			buffer[index++] = norm1.y; //y-normal
			buffer[index++] = norm1.z; //z-normal

			// Vert 2
			buffer[index++] = x * xscale - 1 + xscale; //x
			buffer[index++] = (drawU ? (*this)[y][x].second : (*this)[y][x + 1].first) * yscale; //y
			buffer[index++] = y * zscale - 1; //z

			buffer[index++] = norm1.x; //x-normal
			buffer[index++] = norm1.y; //y-normal
			buffer[index++] = norm1.z; //z-normal

			// Vert 3
			buffer[index++] = x * xscale - 1; //x
			buffer[index++] = (drawU ? (*this)[y][x].second : (*this)[y + 1][x].first) * yscale; //y
			buffer[index++] = y * zscale - 1 + zscale; //z

			buffer[index++] = norm1.x; //x-normal
			buffer[index++] = norm1.y; //y-normal
			buffer[index++] = norm1.z; //z-normal

			// Vert 4
			buffer[index++] = x * xscale - 1 + xscale; //x
			buffer[index++] = (drawU ? (*this)[y][x].second : (*this)[y][x + 1].first) * yscale; //y
			buffer[index++] = y * zscale - 1; //z

			buffer[index++] = norm2.x; //x-normal
			buffer[index++] = norm2.y; //y-normal
			buffer[index++] = norm2.z; //z-normal

			// Vert 5
			buffer[index++] = x * xscale - 1; //x
			buffer[index++] = (drawU ? (*this)[y][x].second : (*this)[y + 1][x].first) * yscale; //y
			buffer[index++] = y * zscale - 1 + zscale; //z

			buffer[index++] = norm2.x; //x-normal
			buffer[index++] = norm2.y; //y-normal
			buffer[index++] = norm2.z; //z-normal

			// Vert 6
			buffer[index++] = x * xscale - 1 + xscale; //x
			buffer[index++] = (drawU ? (*this)[y][x].second : (*this)[y + 1][x + 1].first) * yscale; //y
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

void SimpleGrid::update_ghosts( void (*func)( std::pair<double, double>&, size_t, size_t )){
	for( size_t x = 0; x < x_s; ++x ){
		func( (*this)[0][x], x, 0 );
		func( (*this)[y_s - 1][x], x, y_s - 1 );
	}

	for( size_t y = 1; y < y_s - 1; ++y ){
		func( (*this)[y][0], y, 0 );
		func( (*this)[y][x_s - 1], y, x_s - 1 );
	}
}

void SimpleGrid::step_finite_difference( double dt ){
	for( size_t x = 0; x < x_s; ++x ){
		for( size_t y = 0; y < y_s; ++y ){
			size_t xn = x ? x - 1 : 0;
			size_t yn = y ? y - 1 : 0;
			size_t xp = x != x_s - 1 ? x + 1 : x;
			size_t yp = y != y_s - 1 ? y + 1 : y;

#define IDX( x, y ) ((y) * x_s + (x))
#if 0
			nval[IDX( x, y )] = 2 * values[IDX( x, y)] + C2 *
				( values[IDX(xp, y)] - 2 * values[IDX(x, y)] + values[IDX(xn, y)] +
				  values[IDX(x, yp)] - 2 * values[IDX(x, y)] + values[IDX(x, yn)] ) -
				oval[IDX(x, y)];
#endif
			nval[IDX( x, y )] = std::make_pair<double, double>(
					values[IDX( x, y )].first - K0 * ( values[IDX(xp, y)].second - values[IDX(xn, y)].second +
						values[IDX( x,yp)].second - values[IDX( x,yn)].second ) * 0.5 * dt,
					values[IDX( x, y )].second - onebyrho0 * ( values[IDX(xp, y)].first - values[IDX(xn, y)].first +
						values[IDX( x,yp)].first - values[IDX( x,yn)].first ) * 0.5 * dt
				);
		}
	}

	//std::swap( oval, values );
	std::swap( values, nval );
}

void SimpleGrid::step_finite_volume( double dt ){
	double distance = std::sqrt(K0 * onebyrho0) * dt;

	for( size_t x = 0; x < x_s; ++x ){
		for( size_t y = 0; y < y_s; ++y ){
			size_t xn = x ? x - 1 : 0;
			size_t yn = y ? y - 1 : 0;
			size_t xp = x != x_s - 1 ? x + 1 : x;
			size_t yp = y != y_s - 1 ? y + 1 : y;

			/*
			nval[IDX( x, y )] = std::make_pair<double, double>(
					values[IDX( x, y )].first - K0 * ( values[IDX(xp, y)].second - values[IDX(xn, y)].second +
						values[IDX( x,yp)].second - values[IDX( x,yn)].second ) * 0.5 * dt,
					values[IDX( x, y )].second - onebyrho0 * ( values[IDX(xp, y)].first - values[IDX(xn, y)].first +
						values[IDX( x,yp)].first - values[IDX( x,yn)].first ) * 0.5 * dt
				);
			*/

			nval[IDX( x, y )] = values[IDX(x, y)];

			/*
			//x-1
			auto temp = solveRiemann( x, y, xn, y, dt );
			nval[IDX( x, y )].first -= values[IDX(x, y)].first - values[IDX(x, y)].first * (1 - distance) - temp.first * distance;
			nval[IDX(x, y)].second -= values[IDX(x, y)].second - values[IDX(x, y)].second * (1 - distance) - temp.second * distance;
			//x+1
			temp = solveRiemann( x, y, xp, y, dt );
			if (x == 127 && y == 127) {
				std::cout << values[IDX(x, y)].first << " " << (1 - distance) << " " << temp.first << " " << distance << std::endl;
				std::cout << values[IDX(x, y)].first - values[IDX(x, y)].first * (1 - distance) - temp.first * distance << std::endl;
			}
			nval[IDX(x, y)].first -= values[IDX(x, y)].first - values[IDX(x, y)].first * (1 - distance) - temp.first * distance;
			nval[IDX(x, y)].second -= values[IDX(x, y)].second - values[IDX(x, y)].second * (1 - distance) - temp.second * distance;
			//y-1
			temp = solveRiemann( x, y, x, yn, dt );
			nval[IDX(x, y)].first -= values[IDX(x, y)].first - values[IDX(x, y)].first * (1 - distance) - temp.first * distance;
			nval[IDX(x, y)].second -= values[IDX(x, y)].second - values[IDX(x, y)].second * (1 - distance) - temp.second * distance;
			//y-1
			temp = solveRiemann( x, y, x, yp, dt );
			nval[IDX(x, y)].first -= values[IDX(x, y)].first - values[IDX(x, y)].first * (1 - distance) - temp.first * distance;
			nval[IDX(x, y)].second -= values[IDX(x, y)].second - values[IDX(x, y)].second * (1 - distance) - temp.second * distance;
			*/

			//x-1
			auto temp = solveRiemann(x, y, xn, y, dt);
			nval[IDX(x, y)].first -= (values[IDX(x, y)].first - temp.first) * 0.5;
			nval[IDX(x, y)].second -= (values[IDX(x, y)].second - temp.second) * 0.5;
			//x+1
			temp = solveRiemann(x, y, xp, y, dt);
			if (x == 127 && y == 127) {
				std::cout << values[IDX(x, y)].first << " " << (1 - distance) << " " << temp.first << " " << distance << std::endl;
				std::cout << temp.first / dt << std::endl;
			}
			nval[IDX(x, y)].first -= (values[IDX(x, y)].first - temp.first) * 0.5;
			nval[IDX(x, y)].second -= (values[IDX(x, y)].second - temp.second) * 0.5;
			//y-1
			temp = solveRiemann(x, y, x, yn, dt);
			nval[IDX(x, y)].first -= (values[IDX(x, y)].first - temp.first) * 0.5;
			nval[IDX(x, y)].second -= (values[IDX(x, y)].second - temp.second) * 0.5;
			//y-1
			temp = solveRiemann(x, y, x, yp, dt);
			nval[IDX(x, y)].first -= (values[IDX(x, y)].first - temp.first) * 0.5;
			nval[IDX(x, y)].second -= (values[IDX(x, y)].second - temp.second) * 0.5;

		}
	}

	std::swap( values, nval );


	auto temp = solveRiemann(1, 1, 2, 1, dt);

	std::cout << temp.first << " " << temp.second << std::endl;

}

std::pair<double, double> SimpleGrid::solveRiemann( size_t xl, size_t yl, size_t xr, size_t yr, double dT ){
	/*
	double lambda_min = -std::sqrt( K0 * onebyrho0 );
	double lambda_max = std::sqrt( K0 * onebyrho0 );

	//double q_min_l = 1 / ( lambda_max - lambda_min ) * (lambda_max / (onebyrho0) * (*this)[yl][xl].second - (*this)[yl][xl].first );
	double q_max_l = 1 / ( lambda_min - lambda_max ) * (lambda_min / (onebyrho0) * (*this)[yl][xl].second - (*this)[yl][xl].first );

	double q_min_r = 1 / ( lambda_max - lambda_min ) * (lambda_max / (onebyrho0) * (*this)[yr][xr].second - (*this)[yr][xr].first );
	//double q_max_r = 1 / ( lambda_min - lambda_max ) * (lambda_min / (onebyrho0) * (*this)[yr][xr].second - (*this)[yr][xr].first );

	return std::make_pair<double, double>(
			(( lambda_min * q_min_r * lambda_min ) + ( lambda_max * q_max_l * lambda_max )),
			(( lambda_min * q_min_r * onebyrho0 ) + ( lambda_max * q_max_l * onebyrho0 ))
			);
	*/

	double c = std::sqrt( K0 * onebyrho0 );
	double Z0 = c / onebyrho0;

	double dQp = values[IDX( xr, yr )].first - values[IDX( xl, yl )].first;
	double dQv = values[IDX( xr, yr )].second - values[IDX( xl, yl )].second;

	double alpha1 = (-dQp + dQv * Z0 ) / ( 2 * Z0 );

	double qmp = values[IDX(xl, yl)].first + alpha1 * -Z0;
	double qmv = values[IDX(xl, yl)].second + alpha1 * 1;

	if (xl == 1 && yl == 1 && xr == 1 && yr == 2) {
		std::cout << c << " " << Z0 << " " << dQp << " " << dQv << " " << alpha1 << " " << qmp << " " << qmv << std::endl;
	}

	return { qmp, qmv };
}
