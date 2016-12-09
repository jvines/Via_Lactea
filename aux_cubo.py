"""
Analisis de la via lactea:
Curva de rotacion, perfil de masas y corrugacion del plano galactico.
Jose Vines
"""

import matplotlib.pyplot as plt  # Modulo para graficar
from astropy.io import fits  # Modulo para leer archivos fits
from matplotlib.font_manager import \
    FontProperties  # Funcion para manejar leyendas
from scipy import (argmin, cos, empty_like, fabs, linspace, pi, radians, sin,
                   sqrt, tan, where)
from scipy.optimize import curve_fit  # Funcion para fitear modelos

# GLOBAL VARIABLES

fontP = FontProperties()
fontP.set_size('small')  # Setea el tamano de la letra de las leyendas

R_sun = 2.6228e17  # km
v_sun = 220.  # km/s
G = 6.674e-11 / 1e9  # km^3/kg^2s^2
M_sun = 1e12  # M_sun
radii = 15.  # kpc radio de la via lactea
thick = 0.6  # kpc grosor de la via lactea
s = M_sun / (pi * radii**2)  # kg/kpc^2 densidad superficial de la via lactea
d = M_sun / (1.25 * pi * radii**3)  # kg/kpc^3 densidad de la via lactea


def values(h, j):
    """
    Esta funcion les entrega un arreglo con los valores reales del eje
    correspondiente.
    1 es velocidad, 2 es longitud galactica, 3 es latitud galactica.

    Recibe el header h y el indice j de este.
    """
    N = h['NAXIS' + str(j)]
    val = sp.zeros(N)
    for i in range(0, N):
        val[i] = (i + 1 - float(h['CRPIX' + str(j)])) * \
            float(h['CDELT' + str(j)]) + float(h['CRVAL' + str(j)])
    return val


def corrugation(l, b):
    """Calcula la corrugacion del plano galactico."""
    z = 8.5e3 * cos(l) * tan(b)
    return z


def rotation_curve(v, l):
    """
    Esta funcion calcula la curva de rotacion para una velocidad y longitud
    galactica dada.
    """
    w_sun = v_sun / R_sun
    l_rad = radians(l)
    w = v / (sin(l_rad) * R_sun) + w_sun
    return w


def point_mass_model(m):
    """
    Modelo para la curva de rotacion. Consiste en una masa 'm' puntual

    Parameters
    ----------
    m : float
        Masa del modelo.

    Returns
    -------
    model : array_like
        Modelo de la curva de rotacion.
    """
    model = m
    return model


def uniform_disc_model(r, s):
    """
    Modelo para la curva de rotacion. Consiste en un disco uniforme de densidad
    superficial s

    Parameters
    ----------
    s : float
        Densidad superficial del disco.

    r : array_like
        Vector radio para el modelo.

    Returns
    -------
    model : array_like
        Modelo de la curva de rotacion.
    """
    model = pi * r**2 * s
    return model


def uniform_sphere_model(r, d):
    """
    Modelo para la curva de rotacion. Consiste en una esfera uniforme de
    densidad d

    Parameters
    ----------
    r : array_like
        Vector radio para el modelo.

    d : float
        Densidad superficial de la galaxia.

    Returns
    -------
    model : array_like
        Modelo de la curva de rotacion.
    """
    model = pi * r**3 * d * 1.25
    return model


def point_mass_sphere_model(r, m, d):
    """
    Modelo para la curva de rotacion. Consiste en la suma de una masa puntual
    'm' con una esfera uniforme de densidad d.

    Parameters
    ----------
    r : array_like
        Vector radio para el modelo.

    m : float
        Masa del modelo.

    d : float
        Densidad de la galaxia.

    Returns
    -------
    model : array_like
        Modelo de la curva de rotacion.
    """
    model = m + uniform_sphere_model(r, d)
    return model


def point_mass_disc_model(r, m, s):
    """
    Modelo para la curva de rotacion. Consiste en la suma de una masa puntual
    'm' con un disco uniforme de densidad superficial s.

    Parameters
    ----------
    r : array_like
        Vector radio para el modelo.

    m : float
        Masa del modelo.

    d : float
        Densidad superficial de la galaxia.

    Returns
    -------
    model : array_like
        Modelo de la curva de rotacion.
    """
    model = m + uniform_disc_model(r, s)
    return model


def mass_profile(r, model):
    """Modelo del perfil de masa."""
    model = sqrt(G * model / r)
    return model


def mass_profile_point(r, m):
    """Modelo de perfil de masa de una masa puntual."""
    return mass_profile(r, point_mass_model(m))


def mass_profile_disc(r, s):
    """Modelo de perfil de masa de un disco uniforme."""
    return mass_profile(r, uniform_disc_model(r, s))


def mass_profile_sphere(r, d):
    """Modelo de perfil de masa de una esfera uniforme."""
    return mass_profile(r, uniform_sphere_model(r, d))


def mass_profile_point_disc(r, m, s):
    """Modelo de perfil de masa de una masa puntual mas un disco uniforme."""
    return mass_profile(r, point_mass_disc_model(r, m, s))


def mass_profile_point_sphere(r, m, d):
    """Modelo de perfil de masa de una masa puntual mas una esfera uniforme."""
    return mass_profile(r, point_mass_sphere_model(r, m, d))

if __name__ == '__main__':

    data_cube = "Cubo de datos.fits"
    cubo = fits.open(data_cube)  # abrir objeto cubo de datos.
    data = cubo[0].data  # extraer matriz de datos.
    header = cubo[0].header  # extraer el header del archivo fits.

    # Estos seran los tres arreglos con los valores reales de los tres ejes
    # del cubo.
    vel = values(header, 1)  # vlsr
    lon = values(header, 2)  # l
    lat = values(header, 3)  # b

    l_rad = radians(lon)
    sin_l = -sin(l_rad)

    # Grafico velocidad terminal

    # Crea una lista en donde se guardaran las velocidades minimas (o maximas
    # en modulo) y las velocidades angulares para la curva de rotacion.
    v_terminal = empty_like(lon)
    b = empty_like(lon)

    for l in range(lon.shape[0]):
        idx = where(data[:, l, :] >= .8)
        v_terminal[l] = min(vel[idx[1]])
        idx_b = where(vel[idx[1]] == v_terminal[l])[0][0]
        b[l] = lat[idx[0][idx_b]]

    b_rad = radians(b)

    fig, ax1 = plt.subplots()
    ax1.plot(lon, v_terminal, '.', color='white')
    ax1.set_xlabel('longitud')
    ax1.set_ylabel('velocidad terminal [km/s]')
    ax1.invert_xaxis()
    ax1.invert_yaxis()

    ax2 = ax1.twiny()
    ax2.step(sin_l, v_terminal, color=[.0, .6, .7])
    ax2.set_xlabel('$-\\sin(l)$', fontsize=14)
    ax2.set_ylabel('velocidad terminal km/s')
    ax1.grid()
    plt.savefig('terminal_velocity.pdf', bbox_inches='tight')

    # Grafico de la curva de rotacion con un fit que corresponde a una recta.

    w_r = rotation_curve(v_terminal, lon)
    v_r = fabs(w_r * sp.sin(radians(lon)) * R_sun)

    fig2, ax3 = plt.subplots()
    ax3.plot(sin_l, v_r, ',', color=[0, 0, 0])
    ax3.axhline(220, 0, 1, color='red')
    ax3.set_xlabel('$-\\sin(l)$', fontsize=14)
    ax3.set_ylabel('Velocidad [km/s]')
    ax3.set_title('Curva de Rotacion')
    ax3.set_ylim([160, 280])
    ax3.set_xlim([0.1, 1])
    ax3.grid()
    plt.savefig('rotation_curve_line_fit.pdf', bbox_inches='tight')

    # Modelo masa puntual.
    p0_mass = [M_sun]
    p0_mass_opt, p0_cov_mass = curve_fit(
        mass_profile_point, sin_l, v_r, p0_mass)

    # Modelo disco.
    p0_disc = [s]
    p0_disc_opt, p0_cov_disc = curve_fit(
        mass_profile_disc, sin_l, v_r, p0_disc)

    # Modelo esfera.
    p0_spere = [d]
    p0_sphere_opt, p0_cov_sphere = curve_fit(
        mass_profile_sphere, sin_l, v_r, p0_spere)

    # Modelo masa + disco.
    p0_point_disc = [M_sun, s]
    p0_point_disc_opt, p0_cov_point_disc = curve_fit(
        mass_profile_point_disc, sin_l, v_r, p0_point_disc)
    m_p_d, s_p_d = p0_point_disc_opt

    # Modelo masa + esfera.
    p0_point_sphere = [M_sun, d]
    p0_point_sphere_opt, p0_cov_point_sphere = curve_fit(
        mass_profile_point_sphere, sin_l, v_r, p0_point_sphere)
    m_p_s, d_p_s = p0_point_sphere_opt

    fig3, ax4 = plt.subplots()
    ax4.plot(sin_l, v_r, ',', color=[0, 0, 0])
    ax4.plot(sin_l, mass_profile_point(
        sin_l, p0_mass_opt), '-.', label='Masa puntual')
    ax4.plot(sin_l, mass_profile_disc(sin_l, p0_disc_opt),
             '-.', label='Disco uniforme')
    ax4.plot(sin_l, mass_profile_sphere(sin_l, p0_sphere_opt),
             '-.', label='Esfera uniforme')
    ax4.plot(sin_l, mass_profile_point_disc(sin_l, m_p_d, s_p_d),
             '-.', label='Masa puntual + disco uniforme')
    ax4.plot(sin_l, mass_profile_point_sphere(sin_l, m_p_s, d_p_s),
             '-.', label='Masa puntual + esfera uniforme')
    ax4.set_xlabel('$-\\sin(l)$', fontsize=14)
    ax4.set_ylabel('Velocidad [km/s]')
    ax4.set_title('Perfil de Masa de la Galaxia')
    plt.legend(loc=4, prop=fontP)
    plt.grid()
    plt.savefig('rotation_curve_mas_prof.pdf', bbox_inches='tight')

    # Corrugacion del plano galactico.

    z = corrugation(l_rad, b_rad)

    fig4, ax5 = plt.subplots()
    ax5.step(sin_l, z, color=[.0, .6, .7])
    ax5.set_xlabel('$-\\sin(l)$', fontsize=14)
    ax5.set_ylabel('z [pc]')
    ax5.set_title('Corrugacion del Plano Galactico')
    ax5.set_ylim([-200, 200])
    ax5.grid()
    plt.savefig('corrugation.pdf', bbox_inches='tight')
