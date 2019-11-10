/**
 * File: BRIEF.cpp
 * Author: Dorian Galvez
 * Date: September 2010
 * Description: implementation of BRIEF (Binary Robust Independent 
 *   Elementary Features) descriptor by 
 *   Michael Calonder, Vincent Lepetitand Pascal Fua
 *   + close binary tests (by Dorian Galvez-Lopez)
 * License: see the LICENSE.txt file
 *
 */

#include "BRIEF.h"
#include "../DUtils/DUtils.h"
#include <boost/dynamic_bitset.hpp>
#include <vector>

using namespace std;
using namespace DVision;

// ----------------------------------------------------------------------------

// assert if the program satisfy the condition, it will continue its operations, else it will just stop 
BRIEF::BRIEF(int nbits, int patch_size, Type type):
  m_bit_length(nbits), m_patch_size(patch_size), m_type(type)
{
  assert(patch_size > 1);
  assert(nbits > 0);
  generateTestPoints();
}

// ----------------------------------------------------------------------------

BRIEF::~BRIEF()
{
}

// ---------------------------------------------------------------------------
/**
 * 1. if treat_image is true, convert the image to gray and apply gaussian blur to the image; else, just use the input image
 * 2. every keypoint in the vector points has a descritor, the descriptor has m_x1.size() bits, every bit is determined by pixel value
 *    at (keypoint.x + m_x1[i], keypoint.y + m_y1[i]) and (keypoint.x + m_x2[i], keypoint.y + m_y2[i])
 */
void BRIEF::compute(const cv::Mat &image, 
    const std::vector<cv::KeyPoint> &points,
    vector<bitset> &descriptors,
    bool treat_image) const
{
  const float sigma = 2.f;
  const cv::Size ksize(9, 9);
  
  cv::Mat im;
  // if treat_image is true, convert the image to gray and apply gaussian blur to the image; else, just use the input image
  if(treat_image)
  {
    cv::Mat aux;
    if(image.depth() == 3)
    {
      cv::cvtColor(image, aux, CV_RGB2GRAY);
    }
    else
    {
      aux = image;
    }

    cv::GaussianBlur(aux, im, ksize, sigma, sigma);
    
  }
  else
  {
    im = image;
  }
  
  // check the image is CV_8UC1 and the image is continuous
  assert(im.type() == CV_8UC1);
  assert(im.isContinuous());
  
  // use im now
  const int W = im.cols;
  const int H = im.rows;
  
  descriptors.resize(points.size()); // every element in the vector descriptor is a bitset object
  std::vector<bitset>::iterator dit;

  /*
   * const iterator means the iterator itself is constant, const_iterator means the pointer points to a const vector
   */
  std::vector<cv::KeyPoint>::const_iterator kit;
  
  int x1, y1, x2, y2;
  
  // the vector descriptors has the same size with vector points
  dit = descriptors.begin();
  for(kit = points.begin(); kit != points.end(); ++kit, ++dit)
  {
    dit->resize(m_bit_length);// m_bit_length is the descriptor length in bits
    dit->reset();
    // the size of m_x1 is the same with the size of descriptor, every bit in the descriptor is determined by the corresponding 
    for(unsigned int i = 0; i < m_x1.size(); ++i) 
    {
      x1 = (int)(kit->pt.x + m_x1[i]);
      y1 = (int)(kit->pt.y + m_y1[i]);
      x2 = (int)(kit->pt.x + m_x2[i]);
      y2 = (int)(kit->pt.y + m_y2[i]);
      // every bit in the descriptor is determined by the pixel value at (kit->pt.x + m_x1[i], kit->pt.y + m_y1[i]) and (kit->pt.x + m_x2[i], kit->pt.y + m_y2[i])
      
      if(x1 >= 0 && x1 < W && y1 >= 0 && y1 < H 
        && x2 >= 0 && x2 < W && y2 >= 0 && y2 < H) // compare the pixel value at (x1, y1) and (x2, y2)
      {
        if( im.ptr<unsigned char>(y1)[x1] < im.ptr<unsigned char>(y2)[x2] )
        {
          dit->set(i);
        }        
      } // if (x,y)_1 and (x,y)_2 are in the image
            
    } // for each (x,y)
  } // for each keypoint
}

// ---------------------------------------------------------------------------

void BRIEF::generateTestPoints()
{  
  m_x1.resize(m_bit_length);
  m_y1.resize(m_bit_length);
  m_x2.resize(m_bit_length);
  m_y2.resize(m_bit_length);

  const float g_mean = 0.f;
  const float g_sigma = 0.2f * (float)m_patch_size;
  const float c_sigma = 0.08f * (float)m_patch_size;
  
  float sigma2;
  if(m_type == RANDOM)
    sigma2 = g_sigma;
  else
    sigma2 = c_sigma;
  
  const int max_v = m_patch_size / 2;
  
  DUtils::Random::SeedRandOnce();
  
  for(int i = 0; i < m_bit_length; ++i)
  {
    int x1, y1, x2, y2;
    
    do
    {
      x1 = DUtils::Random::RandomGaussianValue(g_mean, g_sigma);
    } while( x1 > max_v || x1 < -max_v);
    
    do
    {
      y1 = DUtils::Random::RandomGaussianValue(g_mean, g_sigma);
    } while( y1 > max_v || y1 < -max_v);
    
    float meanx, meany;
    if(m_type == RANDOM)
      meanx = meany = g_mean;
    else
    {
      meanx = x1;
      meany = y1;
    }
    
    do
    {
      x2 = DUtils::Random::RandomGaussianValue(meanx, sigma2);
    } while( x2 > max_v || x2 < -max_v);
    
    do
    {
      y2 = DUtils::Random::RandomGaussianValue(meany, sigma2);
    } while( y2 > max_v || y2 < -max_v);
    
    m_x1[i] = x1;
    m_y1[i] = y1;
    m_x2[i] = x2;
    m_y2[i] = y2;
  }

}

// ----------------------------------------------------------------------------


