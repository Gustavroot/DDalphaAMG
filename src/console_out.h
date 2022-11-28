#ifndef CONSOLE_OUT_H
#define CONSOLE_OUT_H

#include "threading.h"
#include "global_struct.h"

static inline void printf0(char *format, ...)
{
    START_MASTER(no_threading)
    if (g.my_rank == 0 && g.print >= 0)
    {
        va_list argpt;
        va_start(argpt, format);
        vprintf(format, argpt);
#ifdef WRITE_LOGFILE
        vfprintf(g.logfile, format, argpt);
        fflush(g.logfile);
#endif
        va_end(argpt);
        fflush(0);
    }
    END_MASTER(no_threading)
}

static inline void warning0(char *format, ...)
{
    if (g.my_rank == 0 && g.print >= 0)
    {
        printf("\x1b[31mwarning: ");
        va_list argpt;
        va_start(argpt, format);
        vprintf(format, argpt);
#ifdef WRITE_LOGFILE
        vfprintf(g.logfile, format, argpt);
        fflush(g.logfile);
#endif
        va_end(argpt);
        printf("\x1b[0m");
        fflush(0);
    }
}

static inline void error0(char *format, ...)
{
    if (g.my_rank == 0)
    {
        printf("\x1b[31merror: ");
        va_list argpt;
        va_start(argpt, format);
        vprintf(format, argpt);
#ifdef WRITE_LOGFILE
        vfprintf(g.logfile, format, argpt);
        fflush(g.logfile);
#endif
        va_end(argpt);
        printf("\x1b[0m");
        fflush(0);
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
}

static inline void printf00(char *format, ...)
{
    if (g.my_rank == 0 && g.print >= 0)
    {
        va_list argpt;
        va_start(argpt, format);
        vprintf(format, argpt);
#ifdef WRITE_LOGFILE
        vfprintf(g.logfile, format, argpt);
        fflush(g.logfile);
#endif
        va_end(argpt);
        fflush(0);
    }
}

#endif // CONSOLE_OUT_H
