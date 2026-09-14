#version 430 core

/*
 * Scientific vehicle graph vertex shader.
 *
 * This is a presentation ABI only.  Positions and constitutive state arrive
 * from the compiler-owned mechanical graph; the shader does not integrate,
 * constrain, smooth, or otherwise alter physics.
 */

layout(location = 0) in vec3  aPosition;
layout(location = 1) in vec3  aNormal;
layout(location = 2) in float aComponentType;
layout(location = 3) in float aAxialStrain;
layout(location = 4) in float aYieldStrain;
layout(location = 5) in float aFractureStrain;
layout(location = 6) in float aDamageState;
layout(location = 7) in float aJointKind;
layout(location = 8) in float aLaplacianEnergy;
layout(location = 9) in float aConservationResidual;

uniform mat4  uMVP;
uniform mat4  uMV;
uniform float uPointSize = 13.0;

out vec3 vNormalV;
out vec3 vPositionV;
out float vAxialStrain;
out float vYieldStrain;
out float vFractureStrain;
flat out int vComponentType;
flat out int vDamageState;
flat out int vJointKind;
out float vLaplacianEnergy;
out float vConservationResidual;

void main() {
    vec4 position_v = uMV * vec4(aPosition, 1.0);
    vPositionV = position_v.xyz;
    vNormalV = mat3(uMV) * aNormal;
    vAxialStrain = aAxialStrain;
    vYieldStrain = max(abs(aYieldStrain), 1.0e-8);
    vFractureStrain = max(abs(aFractureStrain), vYieldStrain);
    vComponentType = int(aComponentType + 0.5);
    vDamageState = int(aDamageState + 0.5);
    vJointKind = int(aJointKind + 0.5);
    vLaplacianEnergy = max(aLaplacianEnergy, 0.0);
    vConservationResidual = max(aConservationResidual, 0.0);
    gl_PointSize = uPointSize;
    gl_Position = uMVP * vec4(aPosition, 1.0);
}
