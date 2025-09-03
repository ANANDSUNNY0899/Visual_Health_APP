// src/SkeletonModel.jsx
import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Color } from 'three';

export function SkeletonModel({ organStress, onOrganClick }) {
  const lungsRef = useRef();
  const liverRef = useRef();
  const healthy = new Color('#8fcf7f');
  const damaged = new Color('#e04a4a');

  const getColor = (val) => healthy.clone().lerp(damaged, val || 0);

  useFrame(({ clock }) => {
    const glow = ((Math.sin(clock.elapsedTime * 3) + 1) / 2) * 0.8;
    if (lungsRef.current) {
      lungsRef.current.material.emissiveIntensity = glow * (organStress.Lungs || 0);
      liverRef.current.material.emissiveIntensity = glow * (organStress.Liver || 0);
    }
  });

  return (
    <group position={[0, -1.2, 0]} scale={1.2}>
      {/* Skull */}
      <mesh position={[0, 2.4, 0]}>
        <sphereGeometry args={[0.3, 32, 32]} />
        <meshStandardMaterial color="#f0f0f0" />
      </mesh>

      {/* Spine */}
      {[...Array(15)].map((_, i) => (
        <mesh key={i} position={[0, 2.1 - i * 0.12, 0.02 * Math.sin(i * 0.3)]}>
          <cylinderGeometry args={[0.06, 0.06, 0.1, 12]} />
          <meshStandardMaterial color="#d0d0d0" />
        </mesh>
      ))}

      {/* Rib Cage */}
      <group position={[0, 1.6, 0]} rotation={[0.2, 0, 0]}>
        {[...Array(7)].map((_, i) => (
          <group key={i} position={[0, -i * 0.15, 0]}>
            <mesh rotation={[0, 1.0, -0.2]}>
              <torusGeometry args={[0.5 - i * 0.03, 0.03, 12, 24, Math.PI * 0.6]} />
              <meshStandardMaterial color="#e5e5e5" />
            </mesh>
            <mesh rotation={[0, -1.0, 0.2]}>
              <torusGeometry args={[0.5 - i * 0.03, 0.03, 12, 24, Math.PI * 0.6]} />
              <meshStandardMaterial color="#e5e5e5" />
            </mesh>
          </group>
        ))}
      </group>

      {/* Pelvis */}
      <group position={[0, 0.6, 0]}>
        <mesh position={[-0.2, 0, 0]} rotation={[Math.PI / 2, 0, 0.2]}>
          <torusGeometry args={[0.3, 0.05, 16, 32, Math.PI * 1.2]} />
          <meshStandardMaterial color="#f0f0f0" />
        </mesh>
        <mesh position={[0.2, 0, 0]} rotation={[Math.PI / 2, 0, -0.2]}>
          <torusGeometry args={[0.3, 0.05, 16, 32, Math.PI * 1.2]} />
          <meshStandardMaterial color="#f0f0f0" />
        </mesh>
      </group>

      {/* Arms */}
      <mesh position={[-0.6, 1.8, 0]} rotation={[0, 0, 0.3]}>
        <cylinderGeometry args={[0.05, 0.05, 0.7, 8]} />
        <meshStandardMaterial color="#f0f0f0" />
      </mesh>
      <mesh position={[0.6, 1.8, 0]} rotation={[0, 0, -0.3]}>
        <cylinderGeometry args={[0.05, 0.05, 0.7, 8]} />
        <meshStandardMaterial color="#f0f0f0" />
      </mesh>

      {/* Legs */}
      <mesh position={[-0.2, -0.4, 0]} rotation={[0, 0, 0]}>
        <cylinderGeometry args={[0.07, 0.07, 1, 12]} />
        <meshStandardMaterial color="#f0f0f0" />
      </mesh>
      <mesh position={[0.2, -0.4, 0]} rotation={[0, 0, 0]}>
        <cylinderGeometry args={[0.07, 0.07, 1, 12]} />
        <meshStandardMaterial color="#f0f0f0" />
      </mesh>

      {/* Organs */}
      <group position={[0, 1.0, 0]}>
        <mesh
          ref={lungsRef}
          position={[-0.2, 0.1, 0.1]}
          onClick={(e) => { e.stopPropagation(); onOrganClick('Lungs'); }}
        >
          <capsuleGeometry args={[0.12, 0.25, 4, 8]} />
          <meshStandardMaterial color={getColor(organStress.Lungs)} emissive="red" />
        </mesh>
        <mesh
          ref={liverRef}
          position={[0.15, -0.3, 0.05]}
          onClick={(e) => { e.stopPropagation(); onOrganClick('Liver'); }}
        >
          <sphereGeometry args={[0.23, 16, 12]} />
          <meshStandardMaterial color={getColor(organStress.Liver)} emissive="red" />
        </mesh>
      </group>
    </group>
  );
}
