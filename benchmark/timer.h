/**
 * @file timer.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-18
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include <sys/time.h>
#include <cstdlib>

class timer {
  private:
    struct timeval t_begin, t_end;
    double         time_taotal;

  public:
    inline timer() : time_taotal(0.0) { gettimeofday(&t_begin, NULL); };
    inline ~timer(){};

    inline double use() { return use_sec(); };
    inline double total() { return time_taotal + use(); }

    inline void   reset();
    inline double use_sec();
    inline double use_usec();
};


inline void timer::reset(void)
{
    time_taotal += use();
    gettimeofday(&t_begin, NULL);
}


inline double timer::use_usec()
{
    gettimeofday(&t_end, NULL);
    return (double) (t_end.tv_sec - t_begin.tv_sec) * 1.0e+6 +
           (double) (t_end.tv_usec - t_begin.tv_usec);
}

inline double timer::use_sec()
{
    gettimeofday(&t_end, NULL);
    return (double) (t_end.tv_sec - t_begin.tv_sec) +
           (double) (t_end.tv_usec - t_begin.tv_usec) * 1.0e-6;
}