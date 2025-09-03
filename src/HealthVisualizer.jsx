// src/HealthVisualizer.jsx
import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment } from '@react-three/drei';
import { AnatomicalModel } from './AnatomicalModel';

export default function HealthVisualizer({ organStress, onOrganClick }) {
  return (
    <Canvas camera={{ position: [0, 0, 4.5], fov: 50 }}>
      <Suspense fallback={null}> 
        <Environment preset="city" />
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} intensity={1} />
        <AnatomicalModel organStress={organStress} onOrganClick={onOrganClick} />
      </Suspense>
      <OrbitControls target={[0, 0.5, 0]} />
    </Canvas>
  );
}