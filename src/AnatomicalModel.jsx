// src/AnatomicalModel.jsx
import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Image } from '@react-three/drei';
import { Color } from 'three';

export function AnatomicalModel({ organStress, onOrganClick }) {
  const lungsRef = useRef();
  const liverRef = useRef();
  // --- NEW ---
  const heartRef = useRef(); 

  const naturalTint = new Color('#ffffff');
  const damagedTint = new Color('#ff8a8a');
  
  const getOrganTint = (stressValue) => {
    if (stressValue === undefined) stressValue = 0;
    const color = naturalTint.clone();
    color.lerp(damagedTint, stressValue);
    return color;
  };

  useFrame((state) => {
    const time = state.clock.elapsedTime;
    const pulse = (Math.sin(time * 3) + 1) / 2;

    if (lungsRef.current) {
      const lungsGlowIntensity = (organStress.Lungs || 0) * pulse * 1.5;
      lungsRef.current.material.emissiveIntensity = lungsGlowIntensity;
    }
    if (liverRef.current) {
      const liverGlowIntensity = (organStress.Liver || 0) * pulse * 1.5;
      liverRef.current.material.emissiveIntensity = liverGlowIntensity;
    }
    // --- NEW ---
    if (heartRef.current) {
        // We'll make the heart pulse based on time, but get brighter with stress
        const heartPulse = (Math.sin(time * 1.5) + 1) / 2; // Slower, more natural pulse
        const heartGlowIntensity = ((organStress.Heart || 0) * 1.2) + (heartPulse * 0.3);
        heartRef.current.material.emissiveIntensity = heartGlowIntensity;
    }
  });

  return (
    <group position={[0, -0.5, 0]}>
      <mesh>
        <cylinderGeometry args={[0.8, 0.6, 2.5, 16]} />
        <meshStandardMaterial color="#cccccc" transparent opacity={0.15} side={2} />
      </mesh>
      
      {/* Lungs and Liver remain the same */}
      <Image ref={lungsRef} url="./lungs.png" position={[0, 0.7, 0]} scale={1.2} transparent onClick={(e) => { e.stopPropagation(); onOrganClick('Lungs'); }} color={getOrganTint(organStress.Lungs)} emissive={damagedTint} />
      <Image ref={liverRef} url="./liver.jpeg" position={[0.2, -0.1, 0.1]} scale={0.8} transparent onClick={(e) => { e.stopPropagation(); onOrganClick('Liver'); }} color={getOrganTint(organStress.Liver)} emissive={damagedTint} />

      {/* --- NEW HEART MODEL --- */}
      <mesh
        ref={heartRef}
        name="Heart"
        position={[0, 0.4, 0.1]}
        onClick={(e) => { e.stopPropagation(); onOrganClick('Heart'); }}
      >
        {/* Using a sphere as a simple placeholder for the heart */}
        <sphereGeometry args={[0.25, 16, 16]} />
        <meshStandardMaterial 
            color={getOrganTint(organStress.Heart)} 
            emissive={damagedTint} 
        />
      </mesh>
    </group>
  );
}