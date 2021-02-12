using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Diagnostics;
using System.Linq;

public class EnvironmentManager : MonoBehaviour
{
    #region Fields and properties

    VoxelGrid _voxelGrid;
    int _randomSeed = 666;

    bool _showVoids = true;

    #endregion

    #region Unity Standard Methods

    void Start()
    {
        // Initialise the voxel grid
        Vector3Int gridSize = new Vector3Int(64, 10, 64);
        _voxelGrid = new VoxelGrid(gridSize, Vector3.zero, 1, parent: this.transform);

        // Set the random engine's seed
        Random.InitState(_randomSeed);
    }

    void Update()
    {
        // Draw the voxels according to their Function Colors
        DrawVoxels();

        // Use the V key to switch between showing voids
        if (Input.GetKeyDown(KeyCode.V))
        {
            _showVoids = !_showVoids;
        }
    }

    #endregion

    #region Private Methods

    /// <summary>
    /// Draws the voxels according to it's state and Function Corlor
    /// </summary>
    void DrawVoxels()
    {
        foreach (var voxel in _voxelGrid.Voxels)
        {
            if (voxel.IsActive)
            {
                Vector3 pos = (Vector3)voxel.Index * _voxelGrid.VoxelSize + transform.position;
                if (voxel.FColor    ==   FunctionColor.Black)   Drawing.DrawCube(pos, _voxelGrid.VoxelSize, Color.black);
                else if (voxel.FColor == FunctionColor.Red)     Drawing.DrawCube(pos, _voxelGrid.VoxelSize, Color.red);
                else if (voxel.FColor == FunctionColor.Yellow)  Drawing.DrawCube(pos, _voxelGrid.VoxelSize, Color.yellow);
                else if (voxel.FColor == FunctionColor.Green)   Drawing.DrawCube(pos, _voxelGrid.VoxelSize, Color.green);
                else if (voxel.FColor == FunctionColor.Cyan)    Drawing.DrawCube(pos, _voxelGrid.VoxelSize, Color.cyan);
                else if (voxel.FColor == FunctionColor.Magenta) Drawing.DrawCube(pos, _voxelGrid.VoxelSize, Color.magenta);
                else if (_showVoids && voxel.Index.y == 0)
                    Drawing.DrawTransparentCube(pos, _voxelGrid.VoxelSize);
            }
        }
    }

    #endregion
}
