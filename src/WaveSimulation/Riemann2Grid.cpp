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

	int res = 4;

	x_s = width / res;
	y_s = height / res;

	values.resize(x_s * y_s);

	for (size_t i = 0; i < x_s * y_s; ++i) {
		values[i].p = glm::vec4(data[i * res] / 255.0);
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

template<typename T>
static T interp0(T w1, T w2) {
	float gauss_legendre[2] = { -0.5f / std::sqrt(3.0f) + 0.5f, 0.5f / std::sqrt(3.0f) + 0.5f };

	return (-gauss_legendre[0] / (gauss_legendre[1] - gauss_legendre[0]) * w2) + (-gauss_legendre[1] / (gauss_legendre[0] - gauss_legendre[1]) * w1);
}

template<typename T>
static T interp1(T w1, T w2) {
	float gauss_legendre[2] = { -0.5f / std::sqrt(3.0f) + 0.5f, 0.5f / std::sqrt(3.0f) + 0.5f };

	return ((1-gauss_legendre[0]) / (gauss_legendre[1] - gauss_legendre[0]) * w2) + ((1-gauss_legendre[1]) / (gauss_legendre[0] - gauss_legendre[1]) * w1);
}

template<typename T>
static T gaussbase0(T w1, float coord) {
	float gauss_legendre[2] = { -0.5f / std::sqrt(3.0f) + 0.5f, 0.5f / std::sqrt(3.0f) + 0.5f };

	return ((coord-gauss_legendre[1]) / (gauss_legendre[0] - gauss_legendre[1]) * w1);
}

template<typename T>
static T gaussbase1(T w1, float coord) {
	float gauss_legendre[2] = { -0.5f / std::sqrt(3.0f) + 0.5f, 0.5f / std::sqrt(3.0f) + 0.5f };

	return ((coord-gauss_legendre[0]) / (gauss_legendre[1] - gauss_legendre[0]) * w1);
}

void Riemann2Grid::fill_buffer(float* buffer, bool drawU) {
	constexpr float yscale = 0.3;
	const float xscale = 2.0 / x_s;
	const float zscale = 2.0 / y_s;

	size_t index = 0;

	for (size_t y = 0; y < y_s - 1; ++y) {
		for (size_t x = 0; x < x_s - 1; ++x) {
			float x1, x2, x3, x4;

			float dx1, dy1, dx2, dy2;
			if (drawU) {
				x1 = interp0( interp0( (*this)[y][x].ux.x, (*this)[y][x].ux.y), interp0((*this)[y][x].ux.z, (*this)[y][x].ux.w ));
				x2 = interp0( interp1( (*this)[y][x].ux.x, (*this)[y][x].ux.y), interp1((*this)[y][x].ux.z, (*this)[y][x].ux.w ));
				x3 = interp1( interp0( (*this)[y][x].ux.x, (*this)[y][x].ux.y), interp0((*this)[y][x].ux.z, (*this)[y][x].ux.w ));
				x4 = interp1( interp1( (*this)[y][x].ux.x, (*this)[y][x].ux.y), interp1((*this)[y][x].ux.z, (*this)[y][x].ux.w ));
			}
			else {
				/*
				x1 = (*this)[y][x].p.x;
				x2 = (*this)[y][x].p.y;
				x3 = (*this)[y][x].p.z;
				x4 = (*this)[y][x].p.w;
				*/
	
				x1 = interp0( interp0( (*this)[y][x].p.x, (*this)[y][x].p.y), interp0((*this)[y][x].p.z, (*this)[y][x].p.w ));
				x2 = interp0( interp1( (*this)[y][x].p.x, (*this)[y][x].p.y), interp1((*this)[y][x].p.z, (*this)[y][x].p.w ));
				x3 = interp1( interp0( (*this)[y][x].p.x, (*this)[y][x].p.y), interp0((*this)[y][x].p.z, (*this)[y][x].p.w ));
				x4 = interp1( interp1( (*this)[y][x].p.x, (*this)[y][x].p.y), interp1((*this)[y][x].p.z, (*this)[y][x].p.w ));

			}

			dx1 = x2 - x1;
			dy1 = x3 - x1;

			dx2 = x4 - x3;
			dy2 = x4 - x2;

			glm::vec3 norm1 = -glm::normalize(glm::vec3{ dx1 / yscale, -1, dy1 / yscale });
			glm::vec3 norm2 = -glm::normalize(glm::vec3{ dx2 / yscale, -1, dy2 / yscale });

			// Vert 1
			buffer[index++] = x * xscale - 1; //x
			buffer[index++] = x1 * yscale; //y
			buffer[index++] = y * zscale - 1; //z

			buffer[index++] = norm1.x; //x-normal
			buffer[index++] = norm1.y; //y-normal
			buffer[index++] = norm1.z; //z-normal

			// Vert 2
			buffer[index++] = x * xscale - 1 + xscale; //x
			buffer[index++] = x2 * yscale; //y
			buffer[index++] = y * zscale - 1; //z

			buffer[index++] = norm1.x; //x-normal
			buffer[index++] = norm1.y; //y-normal
			buffer[index++] = norm1.z; //z-normal

			// Vert 3
			buffer[index++] = x * xscale - 1; //x
			buffer[index++] = x3 * yscale; //y
			buffer[index++] = y * zscale - 1 + zscale; //z

			buffer[index++] = norm1.x; //x-normal
			buffer[index++] = norm1.y; //y-normal
			buffer[index++] = norm1.z; //z-normal

			// Vert 4
			buffer[index++] = x * xscale - 1 + xscale; //x
			buffer[index++] = x2 * yscale; //y
			buffer[index++] = y * zscale - 1; //z

			buffer[index++] = norm2.x; //x-normal
			buffer[index++] = norm2.y; //y-normal
			buffer[index++] = norm2.z; //z-normal

			// Vert 5
			buffer[index++] = x * xscale - 1; //x
			buffer[index++] = x3 * yscale; //y
			buffer[index++] = y * zscale - 1 + zscale; //z

			buffer[index++] = norm2.x; //x-normal
			buffer[index++] = norm2.y; //y-normal
			buffer[index++] = norm2.z; //z-normal

			// Vert 6
			buffer[index++] = x * xscale - 1 + xscale; //x
			buffer[index++] = x4 * yscale; //y
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

void Riemann2Grid::step_finite_volume(double dt) {
	for (size_t x = 0; x < x_s; ++x) {
		for (size_t y = 0; y < y_s; ++y) {
			size_t xn = x ? x - 1 : 0;
			size_t yn = y ? y - 1 : 0;
			size_t xp = x != x_s - 1 ? x + 1 : x;
			size_t yp = y != y_s - 1 ? y + 1 : y;

			nval[IDX(x, y)] = values[IDX(x, y)];

			auto& curr = values[IDX(x, y)];

			glm::mat4 M_inv_p = {
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1
			};

			glm::mat4 M_inv_ux = M_inv_p, M_inv_uy = M_inv_p;

			glm::vec4 face_int_p(0);
			glm::vec4 face_int_ux(0);
			glm::vec4 face_int_uy(0);
			glm::vec3 temp;

			//x-1
			glm::vec3 curr_cell = (interp0(glm::vec3(curr.p.x, curr.ux.x, curr.uy.x), glm::vec3(curr.p.y, curr.ux.y, curr.uy.y)) +
				interp0(glm::vec3(curr.p.z, curr.ux.z, curr.uy.z), glm::vec3(curr.p.w, curr.ux.w, curr.uy.w))) * 0.5f;

			if (xn != x) {
				auto& other = values[IDX(xn, y)];
				glm::vec3 other_cell = (interp1(glm::vec3(other.p.x, other.ux.x, other.uy.x), glm::vec3(other.p.y, other.ux.y, other.uy.y)) +
					interp1(glm::vec3(other.p.z, other.ux.z, other.uy.z), glm::vec3(other.p.w, other.ux.w, other.uy.w))) * 0.5f;

				temp = solveRiemann(curr_cell, other_cell, dt, glm::vec2(-1, 0));
				face_int_p += glm::vec4{ gaussbase0(temp, 0).x, gaussbase1(temp, 0).x, gaussbase0(temp, 0).x, gaussbase1(temp, 0).x };
				face_int_ux += glm::vec4{ gaussbase0(temp, 0).y, gaussbase1(temp, 0).y, gaussbase0(temp, 0).y, gaussbase1(temp, 0).y };
				face_int_uy += glm::vec4{ gaussbase0(temp, 0).z, gaussbase1(temp, 0).z, gaussbase0(temp, 0).z, gaussbase1(temp, 0).z };
			}

			//x+1
			curr_cell = (interp1(glm::vec3(curr.p.x, curr.ux.x, curr.uy.x), glm::vec3(curr.p.y, curr.ux.y, curr.uy.y)) +
				interp1(glm::vec3(curr.p.z, curr.ux.z, curr.uy.z), glm::vec3(curr.p.w, curr.ux.w, curr.uy.w))) * 0.5f;

			if (xp != x) {
				auto& other = values[IDX(xp, y)];
				glm::vec3 other_cell = (interp0(glm::vec3(other.p.x, other.ux.x, other.uy.x), glm::vec3(other.p.y, other.ux.y, other.uy.y)) +
					interp0(glm::vec3(other.p.z, other.ux.z, other.uy.z), glm::vec3(other.p.w, other.ux.w, other.uy.w))) * 0.5f;

				temp = solveRiemann(curr_cell, other_cell, dt, glm::vec2(1, 0));
				face_int_p += glm::vec4{ gaussbase0(temp, 1).x, gaussbase1(temp, 1).x, gaussbase0(temp, 1).x, gaussbase1(temp, 1).x };
				face_int_ux += glm::vec4{ gaussbase0(temp, 1).y, gaussbase1(temp, 1).y, gaussbase0(temp, 1).y, gaussbase1(temp, 1).y };
				face_int_uy += glm::vec4{ gaussbase0(temp, 1).z, gaussbase1(temp, 1).z, gaussbase0(temp, 1).z, gaussbase1(temp, 1).z };
			}
			//y-1
			curr_cell = (interp0(glm::vec3(curr.p.x, curr.ux.x, curr.uy.x), glm::vec3(curr.p.z, curr.ux.z, curr.uy.z)) +
				interp0(glm::vec3(curr.p.y, curr.ux.y, curr.uy.y), glm::vec3(curr.p.w, curr.ux.w, curr.uy.w))) * 0.5f;

			if (yn != y) {
				auto& other = values[IDX(x, yn)];
				glm::vec3 other_cell = (interp1(glm::vec3(other.p.x, other.ux.x, other.uy.x), glm::vec3(other.p.z, other.ux.z, other.uy.z)) +
					interp1(glm::vec3(other.p.y, other.ux.y, other.uy.y), glm::vec3(other.p.w, other.ux.w, other.uy.w))) * 0.5f;

				temp = solveRiemann(curr_cell, other_cell, dt, glm::vec2(0, -1));
				face_int_p += glm::vec4{ gaussbase0(temp, 0).x, gaussbase0(temp, 0).x, gaussbase1(temp, 0).x, gaussbase1(temp, 0).x };
				face_int_ux += glm::vec4{ gaussbase0(temp, 0).y, gaussbase0(temp, 0).y, gaussbase1(temp, 0).y, gaussbase1(temp, 0).y };
				face_int_uy += glm::vec4{ gaussbase0(temp, 0).z, gaussbase0(temp, 0).z, gaussbase1(temp, 0).z, gaussbase1(temp, 0).z };
			}
			//y+1
			curr_cell = (interp1(glm::vec3(curr.p.x, curr.ux.x, curr.uy.x), glm::vec3(curr.p.z, curr.ux.z, curr.uy.z)) +
				interp1(glm::vec3(curr.p.y, curr.ux.y, curr.uy.y), glm::vec3(curr.p.w, curr.ux.w, curr.uy.w))) * 0.5f;

			if (yp != y) {
				auto& other = values[IDX(x, yn)];
				glm::vec3 other_cell = (interp0(glm::vec3(other.p.x, other.ux.x, other.uy.x), glm::vec3(other.p.z, other.ux.z, other.uy.z)) +
					interp0(glm::vec3(other.p.y, other.ux.y, other.uy.y), glm::vec3(other.p.w, other.ux.w, other.uy.w))) * 0.5f;

				temp = solveRiemann(curr_cell, other_cell, dt, glm::vec2(0, 1));
				face_int_p += glm::vec4{ gaussbase0(temp, 1).x, gaussbase0(temp, 1).x, gaussbase1(temp, 1).x, gaussbase1(temp, 1).x };
				face_int_ux += glm::vec4{ gaussbase0(temp, 1).y, gaussbase0(temp, 1).y, gaussbase1(temp, 1).y, gaussbase1(temp, 1).y };
				face_int_uy += glm::vec4{ gaussbase0(temp, 1).z, gaussbase0(temp, 1).z, gaussbase1(temp, 1).z, gaussbase1(temp, 1).z };
			}

			constexpr double face_fac = 2;

			face_int_p *= face_fac;
			face_int_ux *= face_fac;
			face_int_uy *= face_fac;

			//vol int
			float wdev = 1.732050807568877 * 0.5;
			glm::mat4 dev_mat(
				-wdev, wdev, 0, 0,
				wdev, -wdev, 0, 0,
				0, 0, -wdev, wdev,
				0, 0, wdev, -wdev);

			glm::mat3 F =
				glm::mat3(
					0, K0, 0,
					onebyrho0, 0, 0,
					0, 0, 0);

			/*
			glm::vec3 f1 = (F * glm::vec3(curr.p.x, curr.ux.x, curr.uy.x));
			glm::vec3 f2 = (F * glm::vec3(curr.p.y, curr.ux.y, curr.uy.y));
			glm::vec3 f3 = (F * glm::vec3(curr.p.z, curr.ux.z, curr.uy.z));
			glm::vec3 f4 = (F * glm::vec3(curr.p.w, curr.ux.w, curr.uy.w));

			glm::vec4 vol_int_p = dev_mat * glm::vec4(f1.x, f2.x, f3.x, f4.x);
			glm::vec4 vol_int_ux = dev_mat * glm::vec4(f1.y, f2.y, f3.y, f4.y);
			glm::vec4 vol_int_uy = dev_mat * glm::vec4(f1.z, f2.z, f3.z, f4.z);

			vol_int_p.x *= -1;
			vol_int_p.z *= -1;
			vol_int_ux.y *= -1;
			vol_int_ux.w *= -1;
			*/

			/*
			auto delp = dev_mat * curr.p;
			auto delux = dev_mat * curr.ux;
			auto deluy = dev_mat * curr.uy;

			glm::vec3 inte{
				delp.x + delp.z,
				delux.x + delux.z,
				deluy.x + deluy.z
			};

			inte *= 0.5;

			inte = F * inte;

			glm::vec4 vol_int_p{};
			glm::vec4 vol_int_ux{};
			glm::vec4 vol_int_uy{};

			vol_int_p += glm::vec4{ inte.x, inte.x, inte.x, inte.x };
			vol_int_ux += glm::vec4{ inte.y, inte.y, inte.y, inte.y };
			vol_int_uy += glm::vec4{ inte.z, inte.z, inte.z, inte.z };
			*/

			glm::vec3 left_int{( curr.p.x + curr.p.z ), ( curr.ux.x + curr.ux.z ), ( curr.uy.x + curr.uy.z ) };
			glm::vec3 right_int{( curr.p.y + curr.p.w ), ( curr.ux.y + curr.ux.w ), ( curr.uy.y + curr.uy.w ) };

			//left_int *= 0.5f;
			//right_int *= 0.5f;

			auto Fm = F * left_int;
			auto Fp = F * right_int;

			Fp.y *= -1;
			Fp.z *= -1;

			auto res = -1.0f * wdev * ( Fm + Fp );

			glm::vec4 vol_int_p{ -res.x, res.x, -res.x, res.x };
			glm::vec4 vol_int_ux{ res.y, res.y, res.y, res.y };
			glm::vec4 vol_int_uy{ res.z, res.z, res.z, res.z };

			nval[IDX(x, y)].p += (M_inv_p * (
				face_int_p
				+ vol_int_p
				)) * (float)dt;
			nval[IDX(x, y)].ux += (M_inv_ux * (
				face_int_ux
				+ vol_int_ux
				)) * (float)dt;
			nval[IDX(x, y)].uy += (M_inv_uy * (
				face_int_uy
				+ vol_int_uy
				)) * (float)dt;

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

	Fm.y *= -1;
	Fm.z *= -1;

	auto upwind = -0.5f * c * (left - right);

	if (normal.x == 0.0f)
		upwind.y = 0;
	else
		upwind.z = 0;

	return 0.5f * (Fm + Fp) + upwind;
}
