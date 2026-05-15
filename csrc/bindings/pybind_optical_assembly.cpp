/**
 * pybind_optical_assembly.cpp — Python bindings for OpticalAssembly and handlers.
 *
 * Exposes C++ optical_assembly interfaces to Python via pybind11,
 * allowing OpticalAssembly data structures created in Python
 * (optical_assembly.py) to be passed to C++ ray tracing kernels.
 */

#include "optical_handlers.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

/* ──────────────────────────────────────────────────────────────────────────
 * Python Bindings for Optical Components
 * ────────────────────────────────────────────────────────────────────────── */

PYBIND11_MODULE(optical_assembly_bindings, m) {
    m.doc() = "OpticalAssembly and handler bindings for spectral ray tracer";
    
    /* ── OpticalHandlerRole Enum ── */
    py::enum_<OpticalHandlerRole>(m, "OpticalHandlerRole", py::arithmetic())
        .value("STOP", OPTICAL_ROLE_STOP)
        .value("MIRROR", OPTICAL_ROLE_MIRROR)
        .value("REFRACTOR", OPTICAL_ROLE_REFRACTOR)
        .value("FILTER", OPTICAL_ROLE_FILTER)
        .value("DIFFUSER", OPTICAL_ROLE_DIFFUSER)
        .value("SENSOR", OPTICAL_ROLE_SENSOR)
        .value("WAVE_REGION", OPTICAL_ROLE_WAVE_REGION)
        .export_values();
    
    /* ── OpticalRayState Structure ── */
    py::class_<OpticalRayState>(m, "OpticalRayState")
        .def(py::init<>())
        .def_readwrite("pos", &OpticalRayState::pos)
        .def_readwrite("dir", &OpticalRayState::dir)
        .def_readwrite("amplitude", &OpticalRayState::amplitude)
        .def_readwrite("phase_error", &OpticalRayState::phase_error)
        .def_readwrite("wavelength_m", &OpticalRayState::wavelength_m)
        .def_readwrite("cumulative_path_length_m", &OpticalRayState::cumulative_path_length_m)
        .def_readwrite("bounce_count", &OpticalRayState::bounce_count)
        .def_readwrite("is_active", &OpticalRayState::is_active)
        .def_readwrite("hit_sensor", &OpticalRayState::hit_sensor);
    
    /* ── OpticalGeometry Structure ── */
    py::class_<OpticalGeometry>(m, "OpticalGeometry")
        .def(py::init<>())
        .def_readwrite("center_z_m", &OpticalGeometry::center_z_m)
        .def_readwrite("diameter_m", &OpticalGeometry::diameter_m)
        .def_readwrite("radius_m", &OpticalGeometry::radius_m)
        .def_readwrite("radius_of_curvature_m", &OpticalGeometry::radius_of_curvature_m)
        .def_readwrite("conic_k", &OpticalGeometry::conic_k)
        .def_readwrite("aspheric_a4", &OpticalGeometry::aspheric_a4)
        .def_readwrite("aspheric_a6", &OpticalGeometry::aspheric_a6)
        .def_readwrite("surface_roughness_m", &OpticalGeometry::surface_roughness_m);
    
    /* ── OpticalMaterial Structure ── */
    py::class_<OpticalMaterial>(m, "OpticalMaterial")
        .def(py::init<>())
        .def_readwrite("n_real", &OpticalMaterial::n_real)
        .def_readwrite("absorption_coeff_per_m", &OpticalMaterial::absorption_coeff_per_m)
        .def_readwrite("fresnel_r_amplitude", &OpticalMaterial::fresnel_r_amplitude)
        .def_readwrite("fresnel_r_phase", &OpticalMaterial::fresnel_r_phase)
        .def_readwrite("dn_dT_per_K", &OpticalMaterial::dn_dT_per_K);
    
    /* ── OpticalHandlerResult Structure ── */
    py::class_<OpticalHandlerResult>(m, "OpticalHandlerResult")
        .def(py::init<>())
        .def_readwrite("status", &OpticalHandlerResult::status)
        .def_readwrite("rays_blocked", &OpticalHandlerResult::rays_blocked)
        .def_readwrite("rays_hit_surface", &OpticalHandlerResult::rays_hit_surface)
        .def_readwrite("rays_refracted", &OpticalHandlerResult::rays_refracted)
        .def_readwrite("rays_reflected", &OpticalHandlerResult::rays_reflected)
        .def_readwrite("rays_absorbed", &OpticalHandlerResult::rays_absorbed)
        .def_readwrite("rays_transmitted", &OpticalHandlerResult::rays_transmitted);
    
    /* ── OpticalHandler Class (opaque handle) ── */
    py::class_<OpticalHandler, std::unique_ptr<OpticalHandler>>(m, "OpticalHandler")
        .def(py::init<>());
    
    /* ── Handler Factory Functions ── */
    m.def("optical_handler_stop_create", &optical_handler_stop_create,
        py::return_value_policy::reference,
        "Create a STOP (aperture) handler");
    
    m.def("optical_handler_mirror_create", &optical_handler_mirror_create,
        py::return_value_policy::reference,
        "Create a MIRROR (reflection) handler");
    
    m.def("optical_handler_refractor_create", &optical_handler_refractor_create,
        py::return_value_policy::reference,
        "Create a REFRACTOR (lens) handler");
    
    m.def("optical_handler_filter_create", &optical_handler_filter_create,
        py::return_value_policy::reference,
        "Create a FILTER (wavelength absorption) handler");
    
    m.def("optical_handler_diffuser_create", &optical_handler_diffuser_create,
        py::return_value_policy::reference,
        "Create a DIFFUSER (scattering) handler");
    
    m.def("optical_handler_sensor_create", &optical_handler_sensor_create,
        py::return_value_policy::reference,
        "Create a SENSOR (terminal collection) handler");
    
    m.def("optical_handler_process", &optical_handler_process,
        "Process a ray through a single handler");
    
    m.def("optical_handler_destroy", &optical_handler_destroy,
        "Destroy and free a handler");
    
    /* ── OpticalAssembly Class (opaque handle) ── */
    py::class_<OpticalAssembly, std::unique_ptr<OpticalAssembly>>(m, "OpticalAssembly")
        .def(py::init<>());
    
    /* ── Assembly Factory and Processing ── */
    m.def("optical_assembly_create",
        [](py::list handlers_list) {
            std::vector<OpticalHandler*> handlers;
            for (auto h : handlers_list) {
                handlers.push_back(h.cast<OpticalHandler*>());
            }
            return optical_assembly_create(handlers.data(), handlers.size());
        },
        py::return_value_policy::reference,
        "Create optical assembly from handler list");
    
    m.def("optical_assembly_process", &optical_assembly_process,
        "Process a ray through all handlers in assembly");
    
    m.def("optical_assembly_destroy", &optical_assembly_destroy,
        "Destroy and free an assembly");
}
