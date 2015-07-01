//
//  Entropy.h
//
//
//  Created by David Zhang on 24/06/15.
//
//

#ifndef ____Entropy__
#define ____Entropy__

#include <stdio.h>
//correct filename?
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <math.h>
//#define M_PI    3.14159265358979323846

#endif /* defined(____Entropy__) */


//given a covariance matrix Theta (of a MVN) compute the differential entropy using cholesky decomposition
template <typename TValue, int size>
TValue ComputeEntropy(const Eigen::Matrix<TValue, size, size> & Theta)
{
    //mabey check if Theta is a covariance matrix?
    
    //cholesky decomposition of precision matrx
    auto L = Theta.llt().matrixL();
    
    //calculate det(L)
    auto diag = L(0,0);
    for (int i = 1; i<size; i++) {
        diag *= L(i,i);
    }
    //probably better to pre-compute
    const TValue twoPi      = static_cast<TValue>(2*M_PI);
    const TValue n          = static_cast<TValue>(size);
    const TValue oneHalf    = static_cast<TValue>(0.5);
    
    return oneHalf * n * (log(twoPi) + 1) - log(diag);
}

void outputArray(int* array)
{
    for(int x = std::end(array)-1;x >= 0;x--)
    {
        for(int y = std::end(array[x]);y >= 0;y--)
        {
            std::cout << board[i][j]  << "  ";
        }
        std::cout << std::endl;
    }
}

//need public access to us and ps
// -> implemented in Basic::RTF
template<typename TTraits>
void Entropy_for_each_conditioned_subgraph(const typename TTraits::DataSampler& traindb,
                                           const typename TTraits::UnaryFactorTypeVector& Us,
                                           const typename TTraits::PairwiseFactorTypeVector& Ps)
{
    Eigen::Matrix<TValue,size,size> Theta_j;
    
    const int  cx     = ground.Width(), cy = ground.Height();
    TValue entropies[cx][cy];
    
    for(size_t i = 0; i < traindb.GetImageCount(); ++i)
    {
        const auto prep   = TTraits::Feature::PreProcess(traindb.GetInputImage(i));
        const auto ground = traindb.GetGroundTruthImage(i);
        
        
        for(int y = 0; y < cy; ++y)
        {
            for(int x = 0; x < cx; ++x)
            {
                ConditionedSubgraph<TTraits>(Vector2D<int>(x, y), prep, ground, Us, Ps).ComputePrecisionMatrix(Theta_j);
                entropies[x][y] = ComputeEntropy(Theta_j);
                
            }
        }
    }
}
