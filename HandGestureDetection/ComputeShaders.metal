/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Metal shaders used for this sample
*/

#include <metal_stdlib>
using namespace metal;

// Kernel function
kernel void
identityKernel(texture2d<float, access::read_write>  sourceTexture  [[texture(0)]],
               uint2                          gid         [[thread_position_in_grid]])
{
    float pixelVal = sourceTexture.read(gid).r;
    sourceTexture.write(pixelVal, gid);
}

kernel void
softmaxKernel(texture2d<float, access::read>  bgTexture  [[texture(0)]],
             texture2d<float, access::read>  fgTexture  [[texture(1)]],
             texture2d<float, access::read_write> outTexture [[texture(2)]],
             uint2                          gid         [[thread_position_in_grid]])
{
    if((gid.x >= bgTexture.get_width()) || (gid.y >= bgTexture.get_height()))
    {
        return;
    }
    
    float bgScore = bgTexture.read(gid).r;
    float fgScore = fgTexture.read(gid).r;
    float bgScoreExp = exp(bgScore);
    float fgScoreExp = exp(fgScore);
    float denom = bgScoreExp + fgScoreExp;
    float fgScoreSoftmax = fgScoreExp / denom;
    
    outTexture.write(fgScoreSoftmax, gid);
//    outTexture.write(float4(fgScoreSoftmax, 0, 0, 0), gid);
}

kernel void
replaceMaxMinKernel(texture2d<float, access::read> minMaxTexture [[texture(1)]],
                 texture2d<float, access::read> sourceTexture [[texture(2)]],
                 texture2d<float, access::write> outTexture [[texture(0)]],
                                 uint2                          gid         [[thread_position_in_grid]])
{
    if((gid.x >= sourceTexture.get_width()) || (gid.y >= sourceTexture.get_height()))
    {
        return;
    }
    
    float maxProb = minMaxTexture.read(uint2(1, 0)).r;
    float currentProb = sourceTexture.read(gid).r;
    if (currentProb >= maxProb)
    {
        outTexture.write(float4(1, 0, 0, 0), gid);
    }
    else
    {
        outTexture.write(float4(0, 0, 0, 0), gid);
    }
    
}

kernel void
roundKernel(texture2d<float, access::read_write> sourceTexture [[texture(2)]],
            uint2                          gid         [[thread_position_in_grid]])
{
    if((gid.x >= sourceTexture.get_width()) || (gid.y >= sourceTexture.get_height()))
    {
        return;
    }

    float prob = sourceTexture.read(gid).r;
    sourceTexture.write(float4(round(prob), 0, 0, 0), gid);
}

