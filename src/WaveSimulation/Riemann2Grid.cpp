#include "SimpleGrid.hpp"

#include "stb_image.h"
#include <glm/ext/matrix_float3x3.hpp>
#include <glm/geometric.hpp>
#include <iostream>
#include <string.h>

#include <glm/vec3.hpp>
#include <glm/gtx/string_cast.hpp>
#include <utility>
#include <vulkan/vulkan_core.h>

using namespace WaveSimulation;

void Riemann2Grid::init(const char* start_condition) {
	int width, height, channels;

	stbi_uc* data = stbi_load(start_condition, &width, &height, &channels, STBI_grey);

	if (!data) {
		std::cout << "Failed to load texture " << start_condition << " because of: " << stbi_failure_reason() << std::endl;
		return;
	}

	x_s = width;
	y_s = height;

	values.resize(x_s * y_s);

	for (size_t i = 0; i < x_s * y_s; ++i) {
		values[i].p = glm::vec4(data[i] / 255.0);
		values[i].ux = glm::vec4();
		values[i].uy = glm::vec4();
	}

	//oval = values; //Copy
	nval.resize(values.size());

	stbi_image_free(data);
}

size_t Riemann2Grid::get_buffer_float_amount() {
	return (x_s - 1) * (y_s - 1) * 36;
}

void Riemann2Grid::fill_buffer(float* buffer, bool drawU) {
	constexpr float yscale = 0.3;
	const float xscale = 2.0 / x_s;
	const float zscale = 2.0 / y_s;

	size_t index = 0;

	for (size_t y = 0; y < y_s - 1; ++y) {
		for (size_t x = 0; x < x_s - 1; ++x) {
			float dx1, dy1, dx2, dy2;
			if (drawU) {
				dx1 = (*this)[y][x + 1].ux.x - (*this)[y][x].ux.x;
				dy1 = (*this)[y + 1][x].ux.x - (*this)[y][x].ux.x;

				dx2 = (*this)[y + 1][x + 1].ux.x - (*this)[y + 1][x].ux.x;
				dy2 = (*this)[y + 1][x + 1].ux.x - (*this)[y][x + 1].ux.x;
			}
			else {
				dx1 = (*this)[y][x + 1].p.x - (*this)[y][x].p.x;
				dy1 = (*this)[y + 1][x].p.x - (*this)[y][x].p.x;

				dx2 = (*this)[y + 1][x + 1].p.x - (*this)[y + 1][x].p.x;
				dy2 = (*this)[y + 1][x + 1].p.x - (*this)[y][x + 1].p.x;
			}

			glm::vec3 norm1 = -glm::normalize(glm::vec3{ dx1 / yscale, -1, dy1 / yscale });
			glm::vec3 norm2 = -glm::normalize(glm::vec3{ dx2 / yscale, -1, dy2 / yscale });

			// Vert 1
			buffer[index++] = x * xscale - 1; //x
			buffer[index++] = (drawU ? (*this)[y][x].ux.x : (*this)[y][x].p.x) * yscale; //y
			buffer[index++] = y * zscale - 1; //z

			buffer[index++] = norm1.x; //x-normal
			buffer[index++] = norm1.y; //y-normal
			buffer[index++] = norm1.z; //z-normal

			// Vert 2
			buffer[index++] = x * xscale - 1 + xscale; //x
			buffer[index++] = (drawU ? (*this)[y][x].ux.x : (*this)[y][x + 1].p.x) * yscale; //y
			buffer[index++] = y * zscale - 1; //z

			buffer[index++] = norm1.x; //x-normal
			buffer[index++] = norm1.y; //y-normal
			buffer[index++] = norm1.z; //z-normal

			// Vert 3
			buffer[index++] = x * xscale - 1; //x
			buffer[index++] = (drawU ? (*this)[y][x].ux.x : (*this)[y + 1][x].p.x) * yscale; //y
			buffer[index++] = y * zscale - 1 + zscale; //z

			buffer[index++] = norm1.x; //x-normal
			buffer[index++] = norm1.y; //y-normal
			buffer[index++] = norm1.z; //z-normal

			// Vert 4
			buffer[index++] = x * xscale - 1 + xscale; //x
			buffer[index++] = (drawU ? (*this)[y][x].ux.x : (*this)[y][x + 1].p.x) * yscale; //y
			buffer[index++] = y * zscale - 1; //z

			buffer[index++] = norm2.x; //x-normal
			buffer[index++] = norm2.y; //y-normal
			buffer[index++] = norm2.z; //z-normal

			// Vert 5
			buffer[index++] = x * xscale - 1; //x
			buffer[index++] = (drawU ? (*this)[y][x].ux.x : (*this)[y + 1][x].p.x) * yscale; //y
			buffer[index++] = y * zscale - 1 + zscale; //z

			buffer[index++] = norm2.x; //x-normal
			buffer[index++] = norm2.y; //y-normal
			buffer[index++] = norm2.z; //z-normal

			// Vert 6
			buffer[index++] = x * xscale - 1 + xscale; //x
			buffer[index++] = (drawU ? (*this)[y][x].ux.x : (*this)[y + 1][x + 1].p.x) * yscale; //y
			buffer[index++] = y * zscale - 1 + zscale; //z

			buffer[index++] = norm2.x; //x-normal
			buffer[index++] = norm2.y; //y-normal
			buffer[index++] = norm2.z; //z-normal

		}
	}
}

VertexInputDescription Riemann2Grid::get_vk_description() {
	VertexInputDescription desc;

	desc.bindings.emplace_back(VkVertexInputBindingDescription{
			.binding = 0,
			.stride = 6 * sizeof(float),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		});

	desc.attributes.emplace_back(VkVertexInputAttributeDescription{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = 0,
		});

	desc.attributes.emplace_back(VkVertexInputAttributeDescription{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = 3 * sizeof(float),
		});

	return desc;
}

void Riemann2Grid::update_ghosts(void (*func)(Riemann2Cell&, size_t, size_t)) {
	for (size_t x = 0; x < x_s; ++x) {
		func((*this)[0][x], x, 0);
		func((*this)[y_s - 1][x], x, y_s - 1);
	}

	for (size_t y = 1; y < y_s - 1; ++y) {
		func((*this)[y][0], y, 0);
		func((*this)[y][x_s - 1], y, x_s - 1);
	}
}

#define IDX( x, y ) ((y) * x_s + (x))
void Riemann2Grid::step_finite_difference(double dt) {
	/*
	for (size_t x = 0; x < x_s; ++x) {
		for (size_t y = 0; y < y_s; ++y) {
			size_t xn = x ? x - 1 : 0;
			size_t yn = y ? y - 1 : 0;
			size_t xp = x != x_s - 1 ? x + 1 : x;
			size_t yp = y != y_s - 1 ? y + 1 : y;

			//TODO 2d velocity
			nval[IDX(x, y)] = glm::vec3(
				values[IDX(x, y)].x - K0 * (values[IDX(xp, y)].y - values[IDX(xn, y)].y +
					values[IDX(x, yp)].y - values[IDX(x, yn)].y) * 0.5 * dt,
				values[IDX(x, y)].y - onebyrho0 * (values[IDX(xp, y)].x - values[IDX(xn, y)].x +
					values[IDX(x, yp)].x - values[IDX(x, yn)].x) * 0.5 * dt,
				0
			);
		}
	}

	std::swap(values, nval);
	*/
}

template<typename T>
static T interp0(T w1, T w2) {
	float gauss_legendre[2] = { -0.5 / std::sqrt(3) + 0.5, 0.5 / std::sqrt(3) + 0.5 };

	return (-gauss_legendre[0] / (gauss_legendre[1] - gauss_legendre[0]) * w2) + (-gauss_legendre[1] / (gauss_legendre[0] - gauss_legendre[1]) * w1);
}

template<typename T>
static T interp1(T w1, T w2) {
	float gauss_legendre[2] = { -0.5 / std::sqrt(3) + 0.5, 0.5 / std::sqrt(3) + 0.5 };

	return ((1-gauss_legendre[0]) / (gauss_legendre[1] - gauss_legendre[0]) * w2) + ((1-gauss_legendre[1]) / (gauss_legendre[0] - gauss_legendre[1]) * w1);
}

void Riemann2Grid::step_finite_volume(double dt) {
	float distance = std::sqrt(K0 * onebyrho0) * dt * 10;

	for (size_t x = 0; x < x_s; ++x) {
		for (size_t y = 0; y < y_s; ++y) {
			size_t xn = x ? x - 1 : 0;
			size_t yn = y ? y - 1 : 0;
			size_t xp = x != x_s - 1 ? x + 1 : x;
			size_t yp = y != y_s - 1 ? y + 1 : y;

			nval[IDX(x, y)] = values[IDX(x, y)];

			auto& curr = values[IDX(x, y)];

			/*
			glm::mat4 M_inv_p = {
				1 / curr.p.x, 0, 0, 0,
				0, 1 / curr.p.y, 0, 0,
				0, 0, 1 / curr.p.z, 0,
				0, 0, 0, 1 / curr.p.w
			};

			glm::mat4 M_inv_ux = {
				1 / curr.ux.x, 0, 0, 0,
				0, 1 / curr.ux.y, 0, 0,
				0, 0, 1 / curr.ux.z, 0,
				0, 0, 0, 1 / curr.ux.w
			};

			glm::mat4 M_inv_uy = {
				1 / curr.uy.x, 0, 0, 0,
				0, 1 / curr.uy.y, 0, 0,
				0, 0, 1 / curr.uy.z, 0,
				0, 0, 0, 1 / curr.uy.w
			};
			*/

			glm::mat4 M_inv_p = {
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1
			};

			glm::mat4 M_inv_ux = M_inv_p, M_inv_uy = M_inv_p;



			glm::vec3 face_int{ 0, 0, 0 };
			glm::vec3 temp;

			//x-1
			glm::vec3 curr_cell = (interp0(glm::vec3(curr.p.x, curr.ux.x, curr.uy.x), glm::vec3(curr.p.y, curr.ux.y, curr.uy.y)) +
				interp0(glm::vec3(curr.p.z, curr.ux.z, curr.uy.z), glm::vec3(curr.p.w, curr.ux.w, curr.uy.w))) * 0.5f;

			if (xn != x) {
				auto& other = values[IDX(xn, y)];
				glm::vec3 other_cell = (interp1(glm::vec3(other.p.x, other.ux.x, other.uy.x), glm::vec3(other.p.y, other.ux.y, other.uy.y)) +
					interp1(glm::vec3(other.p.z, other.ux.z, other.uy.z), glm::vec3(other.p.w, other.ux.w, other.uy.w))) * 0.5f;

				temp = solveRiemann(curr_cell, other_cell, dt, glm::vec2(-1, 0));
			}
			else {
				temp = solveRiemann(curr_cell, curr_cell, dt, glm::vec2(-1, 0));
			}
			face_int += temp * distance;

			//x+1
			curr_cell = (interp1(glm::vec3(curr.p.x, curr.ux.x, curr.uy.x), glm::vec3(curr.p.y, curr.ux.y, curr.uy.y)) +
				interp1(glm::vec3(curr.p.z, curr.ux.z, curr.uy.z), glm::vec3(curr.p.w, curr.ux.w, curr.uy.w))) * 0.5f;

			if (xp != x) {
				auto& other = values[IDX(xp, y)];
				glm::vec3 other_cell = (interp0(glm::vec3(other.p.x, other.ux.x, other.uy.x), glm::vec3(other.p.y, other.ux.y, other.uy.y)) +
					interp0(glm::vec3(other.p.z, other.ux.z, other.uy.z), glm::vec3(other.p.w, other.ux.w, other.uy.w))) * 0.5f;

				temp = solveRiemann(curr_cell, other_cell, dt, glm::vec2(1, 0));
			}
			else {
				temp = solveRiemann(curr_cell, curr_cell, dt, glm::vec2(1, 0));
			}
			face_int += temp * distance;
			//y-1
			curr_cell = (interp0(glm::vec3(curr.p.x, curr.ux.x, curr.uy.x), glm::vec3(curr.p.z, curr.ux.z, curr.uy.z)) +
				interp0(glm::vec3(curr.p.y, curr.ux.y, curr.uy.y), glm::vec3(curr.p.w, curr.ux.w, curr.uy.w))) * 0.5f;

			if (yn != y) {
				auto& other = values[IDX(x, yn)];
				glm::vec3 other_cell = (interp1(glm::vec3(curr.p.x, curr.ux.x, curr.uy.x), glm::vec3(curr.p.z, curr.ux.z, curr.uy.z)) +
					interp1(glm::vec3(curr.p.y, curr.ux.y, curr.uy.y), glm::vec3(curr.p.w, curr.ux.w, curr.uy.w))) * 0.5f;

				temp = solveRiemann(curr_cell, other_cell, dt, glm::vec2(0, -1));
			}
			else {
				temp = solveRiemann(curr_cell, curr_cell, dt, glm::vec2(0, -1));
			}
			face_int += temp * distance;
			//y+1
			curr_cell = (interp1(glm::vec3(curr.p.x, curr.ux.x, curr.uy.x), glm::vec3(curr.p.z, curr.ux.z, curr.uy.z)) +
				interp1(glm::vec3(curr.p.y, curr.ux.y, curr.uy.y), glm::vec3(curr.p.w, curr.ux.w, curr.uy.w))) * 0.5f;

			if (yp != y) {
				auto& other = values[IDX(x, yn)];
				glm::vec3 other_cell = (interp0(glm::vec3(curr.p.x, curr.ux.x, curr.uy.x), glm::vec3(curr.p.z, curr.ux.z, curr.uy.z)) +
					interp0(glm::vec3(curr.p.y, curr.ux.y, curr.uy.y), glm::vec3(curr.p.w, curr.ux.w, curr.uy.w))) * 0.5f;

				temp = solveRiemann(curr_cell, other_cell, dt, glm::vec2(0, 1));
			}
			else {
				temp = solveRiemann(curr_cell, curr_cell, dt, glm::vec2(0, 1));
			}
			face_int += temp * distance;

			//face_int = { 0, 0, 0 };

			nval[IDX(x, y)].p += M_inv_p * glm::vec4(face_int.x);
			nval[IDX(x, y)].ux += M_inv_ux * glm::vec4(face_int.y);
			nval[IDX(x, y)].uy += M_inv_uy * glm::vec4(face_int.z);
			auto asdf = nval[IDX(x, y)];
		}
	}

	std::swap(values, nval);
}

glm::vec3 Riemann2Grid::solveRiemann(glm::vec3 left, glm::vec3 right, double dT, glm::vec2 normal) {

	float c = std::sqrt(K0 * onebyrho0);

	glm::mat3 F =
		glm::mat3(
			0, K0, 0,
			onebyrho0, 0, 0,
			0, 0, 0) * normal.x +
		glm::mat3(
			0, 0, K0,
			0, 0, 0,
			onebyrho0, 0, 0) * normal.y;

	glm::vec3 Fm = F * left;
	glm::vec3 Fp = F * right;

	auto upwind = -0.5f * c * (left - right);

	if (normal.x == 0.0f)
		upwind.y = 0;
	else
		upwind.z = 0;

	return 0.5f * (Fm + Fp) + upwind;
}
