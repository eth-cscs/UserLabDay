program lammps_init

! gfortran 0.F90 && ./a.out <<< 1000
! nx=          25
! ny=          40

    implicit none
    integer :: i,j,np

    print *,' np=?'
    read (5,*) np

    do i=1,np
    do j=1,np
        if (i*j==np .and. (i-j)>0) then
            print *,i,j,i*j,i-j
            print *,'nx=',j
            print *,'ny=',i
            stop
        endif
    enddo
    enddo

end program lammps_init
