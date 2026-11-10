"""Perfect-diffuse surface reflectance shared by Studio packet/camera paths.

Only opaque, nonemissive, fully diffuse faces use this route. RGB remains the
existing illustrative spectral approximation, not measured reflectance data.
Roulette survival is a sampling probability, never a material reflectance floor.
"""
STUDIO_DIFFUSE_GLSL = r'''
bool studio_diffuse_surface(Tri t) {
    return t.mat_out.w >= 0.999 && t.mat_in.y == 1.0 && t.mat_out.y == 1.0
        && (floatBitsToUint(t.normal.w) & MAT_FLAG_EMISSIVE) == 0u;
}
float studio_diffuse_rho(Tri t, float frequency) {
    return clamp(studio_material_weight(t.albedo.rgb, frequency), 0.0, 1.0);
}
float studio_diffuse_survival(vec3 albedo) {
    return clamp(max(albedo.r,max(albedo.g,albedo.b)), 0.05, 0.98);
}
'''
