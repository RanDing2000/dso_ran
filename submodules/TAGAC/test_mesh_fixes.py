#!/usr/bin/env python3
"""
Test script to verify mesh compatibility fixes
"""

import sys
import os
import numpy as np
import trimesh

# Add project root to path
sys.path.append('/home/ran.ding/projects/TARGO')

def test_open3d_to_trimesh_conversion():
    """Test Open3D to Trimesh conversion"""
    try:
        import open3d as o3d
        from src.utils_targo import save_open3d_mesh
        
        # Create a simple Open3D mesh
        cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        cube.compute_vertex_normals()
        
        # Test saving
        test_path = "/tmp/test_open3d_mesh.obj"
        save_open3d_mesh(cube, test_path)
        
        if os.path.exists(test_path):
            print("✅ Open3D to Trimesh conversion test passed")
            os.remove(test_path)
            return True
        else:
            print("❌ Open3D to Trimesh conversion test failed")
            return False
            
    except Exception as e:
        print(f"❌ Open3D to Trimesh conversion test failed: {e}")
        return False

def test_trimesh_compatibility():
    """Test Trimesh mesh compatibility"""
    try:
        from src.utils_targo import compute_chamfer_and_iou
        
        # Create a simple trimesh
        cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        
        # Create dummy completed point cloud
        completed_pc = np.random.rand(1000, 3) - 0.5
        
        # Test computation
        cd, iou = compute_chamfer_and_iou(cube, completed_pc)
        
        print(f"✅ Trimesh compatibility test passed - CD: {cd:.6f}, IoU: {iou:.6f}")
        return True
        
    except Exception as e:
        print(f"❌ Trimesh compatibility test failed: {e}")
        return False

def test_open3d_mesh_compatibility():
    """Test Open3D mesh compatibility"""
    try:
        import open3d as o3d
        from src.utils_targo import compute_chamfer_and_iou
        
        # Create a simple Open3D mesh
        cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        
        # Create dummy completed point cloud
        completed_pc = np.random.rand(1000, 3) - 0.5
        
        # Test computation
        cd, iou = compute_chamfer_and_iou(cube, completed_pc)
        
        print(f"✅ Open3D mesh compatibility test passed - CD: {cd:.6f}, IoU: {iou:.6f}")
        return True
        
    except Exception as e:
        print(f"❌ Open3D mesh compatibility test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing mesh compatibility fixes...")
    print("=" * 50)
    
    tests = [
        test_open3d_to_trimesh_conversion,
        test_trimesh_compatibility,
        test_open3d_mesh_compatibility
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Mesh compatibility fixes are working.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)





