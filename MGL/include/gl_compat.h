/*
 * gl_compat.h — public declarations and enums of the compatibility profile
 * (compat domain).  Include after the core header to use the fixed-function
 * API; the calls are live only in a compatibility-profile context
 * (see mgl_compat.h).
 */
#ifndef GL_COMPAT_H
#define GL_COMPAT_H
#include <stddef.h>

#include <GL/glcorearb.h>


#ifndef MGL_COMPAT_ENUMS_INCLUDED
#define MGL_COMPAT_ENUMS_INCLUDED
/*
 * mgl_compat_enums.h — compatibility-profile enums missing from glcorearb.h.
 *
 * Values from the GL registry (gl.h / glext.h compatibility tokens).
 */



#ifndef GL_ACCUM
#define GL_ACCUM 0x0100
#define GL_LOAD 0x0101
#define GL_RETURN 0x0102
#define GL_MULT 0x0103
#define GL_ADD 0x0104
#define GL_2D 0x1400
#define GL_3D 0x1402
#define GL_3D_COLOR 0x1403
#define GL_3D_COLOR_TEXTURE 0x1404
#define GL_4D_COLOR_TEXTURE 0x1405
#define GL_2_BYTES 0x1407
#define GL_3_BYTES 0x1408
#define GL_4_BYTES 0x1409
#define GL_EXP 0x0800
#define GL_EXP2 0x0801
#define GL_FOG 0x0B60
#define GL_FOG_INDEX 0x0B61
#define GL_FOG_DENSITY 0x0B62
#define GL_FOG_START 0x0B63
#define GL_FOG_END 0x0B64
#define GL_FOG_COLOR 0x0B66
#define GL_FOG_MODE 0x0B65
#define GL_LIGHTING 0x0B50
#define GL_LIGHT_MODEL_LOCAL_VIEWER 0x0B51
#define GL_LIGHT_MODEL_TWO_SIDE 0x0B52
#define GL_LIGHT_MODEL_AMBIENT 0x0B53
#define GL_LIGHT0 0x4000
#define GL_LIGHT1 0x4001
#define GL_LIGHT2 0x4002
#define GL_LIGHT3 0x4003
#define GL_LIGHT4 0x4004
#define GL_LIGHT5 0x4005
#define GL_LIGHT6 0x4006
#define GL_LIGHT7 0x4007
#define GL_AMBIENT 0x1200
#define GL_DIFFUSE 0x1201
#define GL_SPECULAR 0x1202
#define GL_POSITION 0x1203
#define GL_SPOT_DIRECTION 0x1204
#define GL_SPOT_EXPONENT 0x1205
#define GL_SPOT_CUTOFF 0x1206
#define GL_CONSTANT_ATTENUATION 0x1207
#define GL_LINEAR_ATTENUATION 0x1208
#define GL_QUADRATIC_ATTENUATION 0x1209
#define GL_EMISSION 0x1600
#define GL_SHININESS 0x1601
#define GL_AMBIENT_AND_DIFFUSE 0x1602
#define GL_COLOR_INDEXES 0x1603
#define GL_COLOR_MATERIAL 0x0B57
#define GL_COLOR_MATERIAL_PARAMETER 0x0B56
#define GL_COLOR_MATERIAL_FACE 0x0B55
#define GL_LIGHT_MODEL_COLOR_CONTROL 0x81F8
#define GL_SINGLE_COLOR 0x81F9
#define GL_SEPARATE_SPECULAR_COLOR 0x81FA
#define GL_COLOR_SUM 0x8458
#define GL_NORMALIZE 0x0BA1
#define GL_RESCALE_NORMAL 0x803A
#define GL_AUTO_NORMAL 0x0D80
#define GL_MATRIX_MODE 0x0BA0
#define GL_MODELVIEW 0x1700
#define GL_PROJECTION 0x1701
#define GL_MODELVIEW_MATRIX 0x0BA6
#define GL_PROJECTION_MATRIX 0x0BA7
#define GL_TEXTURE_MATRIX 0x0BA8
#define GL_COLOR_MATRIX 0x80B1
#define GL_MODELVIEW_STACK_DEPTH 0x0BA3
#define GL_PROJECTION_STACK_DEPTH 0x0BA4
#define GL_TEXTURE_STACK_DEPTH 0x0BA5
#define GL_MAX_MODELVIEW_STACK_DEPTH 0x0D36
#define GL_MAX_PROJECTION_STACK_DEPTH 0x0D38
#define GL_MAX_TEXTURE_STACK_DEPTH 0x0D39
#define GL_MAX_LIGHTS 0x0D31
#define GL_MAX_CLIP_PLANES 0x0D32
#define GL_MAX_TEXTURE_COORDS 0x8871
#define GL_MAX_TEXTURE_UNITS 0x84E2
#define GL_MAX_ATTRIB_STACK_DEPTH 0x0D35
#define GL_MAX_CLIENT_ATTRIB_STACK_DEPTH 0x0D3B
#define GL_MAX_NAME_STACK_DEPTH 0x0D37
#define GL_MAX_LIST_NESTING 0x0B31
#define GL_MAX_EVAL_ORDER 0x0D30
#define GL_MAX_PIXEL_MAP_TABLE 0x0D34
#define GL_NAME_STACK_DEPTH 0x0D70
#define GL_CURRENT_COLOR 0x0B00
#define GL_CURRENT_INDEX 0x0B01
#define GL_CURRENT_NORMAL 0x0B02
#define GL_CURRENT_TEXTURE_COORDS 0x0B03
#define GL_CURRENT_RASTER_COLOR 0x0B04
#define GL_CURRENT_RASTER_INDEX 0x0B05
#define GL_CURRENT_RASTER_TEXTURE_COORDS 0x0B06
#define GL_CURRENT_RASTER_POSITION 0x0B07
#define GL_CURRENT_RASTER_POSITION_VALID 0x0B08
#define GL_CURRENT_RASTER_DISTANCE 0x0B09
#define GL_EDGE_FLAG 0x0B44
#define GL_COMPILE 0x1300
#define GL_COMPILE_AND_EXECUTE 0x1301
#define GL_LIST_MODE 0x0B30
#define GL_LIST_INDEX 0x0B33
#define GL_LIST_BASE 0x0B32
#define GL_MAP1_COLOR_4 0x0D90
#define GL_MAP1_INDEX 0x0D91
#define GL_MAP1_NORMAL 0x0D92
#define GL_MAP1_TEXTURE_COORD_1 0x0D93
#define GL_MAP1_TEXTURE_COORD_2 0x0D94
#define GL_MAP1_TEXTURE_COORD_3 0x0D95
#define GL_MAP1_TEXTURE_COORD_4 0x0D96
#define GL_MAP1_VERTEX_3 0x0D97
#define GL_MAP1_VERTEX_4 0x0D98
#define GL_MAP2_COLOR_4 0x0DB0
#define GL_MAP2_INDEX 0x0DB1
#define GL_MAP2_NORMAL 0x0DB2
#define GL_MAP2_TEXTURE_COORD_1 0x0DB3
#define GL_MAP2_TEXTURE_COORD_2 0x0DB4
#define GL_MAP2_TEXTURE_COORD_3 0x0DB5
#define GL_MAP2_TEXTURE_COORD_4 0x0DB6
#define GL_MAP2_VERTEX_3 0x0DB7
#define GL_MAP2_VERTEX_4 0x0DB8
#define GL_COEFF 0x0A00
#define GL_ORDER 0x0A01
#define GL_DOMAIN 0x0A02
#define GL_POLYGON_STIPPLE 0x0B42
#define GL_LINE_STIPPLE 0x0B24
#define GL_LINE_STIPPLE_PATTERN 0x0B25
#define GL_LINE_STIPPLE_REPEAT 0x0B26
#define GL_POLYGON 0x0009
#define GL_QUAD_STRIP 0x0008
#define GL_SMOOTH 0x1D01
#define GL_FLAT 0x1D00
#define GL_VERTEX_ARRAY 0x8074
#define GL_NORMAL_ARRAY 0x8075
#define GL_COLOR_ARRAY 0x8076
#define GL_INDEX_ARRAY 0x8077
#define GL_TEXTURE_COORD_ARRAY 0x8078
#define GL_EDGE_FLAG_ARRAY 0x8079
#define GL_VERTEX_ARRAY_POINTER 0x808E
#define GL_NORMAL_ARRAY_POINTER 0x808F
#define GL_COLOR_ARRAY_POINTER 0x8090
#define GL_INDEX_ARRAY_POINTER 0x8091
#define GL_TEXTURE_COORD_ARRAY_POINTER 0x8092
#define GL_EDGE_FLAG_ARRAY_POINTER 0x8093
#define GL_SECONDARY_COLOR_ARRAY 0x845E
#define GL_SECONDARY_COLOR_ARRAY_POINTER 0x845F
#define GL_FOG_COORD_ARRAY 0x8457
#define GL_FOG_COORD_ARRAY_POINTER 0x8456
#define GL_V2F 0x2A20
#define GL_V3F 0x2A21
#define GL_C4UB_V2F 0x2A22
#define GL_C4UB_V3F 0x2A23
#define GL_C3F_V3F 0x2A24
#define GL_N3F_V3F 0x2A25
#define GL_C4F_N3F_V3F 0x2A26
#define GL_T2F_V3F 0x2A27
#define GL_T4F_V4F 0x2A28
#define GL_T2F_C4UB_V3F 0x2A29
#define GL_T2F_C3F_V3F 0x2A2A
#define GL_T2F_N3F_V3F 0x2A2B
#define GL_T2F_C4F_N3F_V3F 0x2A2C
#define GL_T4F_C4F_N3F_V4F 0x2A2D
#define GL_TEXTURE_ENV 0x2300
#define GL_TEXTURE_ENV_MODE 0x2200
#define GL_TEXTURE_ENV_COLOR 0x2201
#define GL_MODULATE 0x2100
#define GL_DECAL 0x2101
#define GL_COMBINE 0x8570
#define GL_COMBINE_RGB 0x8571
#define GL_COMBINE_ALPHA 0x8572
#define GL_SOURCE0_RGB 0x8580
#define GL_SOURCE1_RGB 0x8581
#define GL_SOURCE2_RGB 0x8582
#define GL_SOURCE0_ALPHA 0x8588
#define GL_SOURCE1_ALPHA 0x8589
#define GL_SOURCE2_ALPHA 0x858A
#define GL_SRC0_RGB 0x8580
#define GL_SRC1_RGB 0x8581
#define GL_SRC2_RGB 0x8582
#define GL_SRC0_ALPHA 0x8588
#define GL_SRC1_ALPHA 0x8589
#define GL_SRC2_ALPHA 0x858A
#define GL_OPERAND0_RGB 0x8590
#define GL_OPERAND1_RGB 0x8591
#define GL_OPERAND2_RGB 0x8592
#define GL_OPERAND0_ALPHA 0x8598
#define GL_OPERAND1_ALPHA 0x8599
#define GL_OPERAND2_ALPHA 0x859A
#define GL_RGB_SCALE 0x8573
#define GL_ADD_SIGNED 0x8574
#define GL_INTERPOLATE 0x8575
#define GL_SUBTRACT 0x84E7
#define GL_CONSTANT 0x8576
#define GL_PRIMARY_COLOR 0x8577
#define GL_PREVIOUS 0x8578
#define GL_DOT3_RGB 0x86AE
#define GL_DOT3_RGBA 0x86AF
#define GL_S 0x2000
#define GL_T 0x2001
#define GL_R 0x2002
#define GL_Q 0x2003
#define GL_TEXTURE_GEN_MODE 0x2500
#define GL_OBJECT_PLANE 0x2501
#define GL_EYE_PLANE 0x2502
#define GL_OBJECT_LINEAR 0x2401
#define GL_EYE_LINEAR 0x2400
#define GL_SPHERE_MAP 0x2402
#define GL_NORMAL_MAP 0x8511
#define GL_REFLECTION_MAP 0x8512
#define GL_TEXTURE_GEN_S 0x0C60
#define GL_TEXTURE_GEN_T 0x0C61
#define GL_TEXTURE_GEN_R 0x0C62
#define GL_TEXTURE_GEN_Q 0x0C63
#define GL_MAP_COLOR 0x0D10
#define GL_MAP_STENCIL 0x0D11
#define GL_INDEX_SHIFT 0x0D12
#define GL_INDEX_OFFSET 0x0D13
#define GL_RED_SCALE 0x0D14
#define GL_RED_BIAS 0x0D15
#define GL_ZOOM_X 0x0D16
#define GL_ZOOM_Y 0x0D17
#define GL_GREEN_SCALE 0x0D18
#define GL_GREEN_BIAS 0x0D19
#define GL_BLUE_SCALE 0x0D1A
#define GL_BLUE_BIAS 0x0D1B
#define GL_ALPHA_SCALE 0x0D1C
#define GL_ALPHA_BIAS 0x0D1D
#define GL_DEPTH_SCALE 0x0D1E
#define GL_DEPTH_BIAS 0x0D1F
#define GL_POST_COLOR_MATRIX_RED_SCALE 0x80B4
#define GL_POST_COLOR_MATRIX_GREEN_SCALE 0x80B5
#define GL_PIXEL_MAP_I_TO_I 0x0C70
#define GL_PIXEL_MAP_S_TO_S 0x0C71
#define GL_PIXEL_MAP_I_TO_R 0x0C72
#define GL_PIXEL_MAP_I_TO_G 0x0C73
#define GL_PIXEL_MAP_I_TO_B 0x0C74
#define GL_PIXEL_MAP_I_TO_A 0x0C75
#define GL_PIXEL_MAP_R_TO_R 0x0C76
#define GL_PIXEL_MAP_G_TO_G 0x0C77
#define GL_PIXEL_MAP_B_TO_B 0x0C78
#define GL_PIXEL_MAP_A_TO_A 0x0C79
#define GL_LUMINANCE 0x1909
#define GL_LUMINANCE_ALPHA 0x190A
#define GL_RENDER 0x1C00
#define GL_FEEDBACK 0x1C01
#define GL_SELECT 0x1C02
#define GL_RENDER_MODE 0x0C40
#define GL_SELECTION_BUFFER_POINTER 0xDF3
#define GL_FEEDBACK_BUFFER_POINTER 0xDF1
#define GL_POLYGON_TOKEN 0x0703
#define GL_LINE_TOKEN 0x0702
#define GL_POINT_TOKEN 0x0701
#define GL_PASS_THROUGH_TOKEN 0x0806
#define GL_POINT_SMOOTH 0x0B10
#define GL_POINT_SIZE_MIN 0x8126
#define GL_POINT_SIZE_MAX 0x8127
#define GL_POINT_DISTANCE_ATTENUATION 0x8129
#define GL_POINT_SPRITE 0x8861
#define GL_COORD_REPLACE 0x8862
#define GL_ALPHA_TEST 0x0BC0
#define GL_ALPHA_TEST_REF 0x0BC2
#define GL_ALPHA_TEST_FUNC 0x0BC1
#define GL_INDEX_MODE 0x0C30
#define GL_RGBA_MODE 0x0C31
#define GL_AUX_BUFFERS 0x0C00
#define GL_INDEX_LOGIC_OP 0x0C1F
#define GL_SHADE_MODEL 0x0B54
#define GL_CLAMP 0x2900
#define GL_GENERATE_MIPMAP 0x8291
#define GL_CLIENT_ACTIVE_TEXTURE 0x84E1
#define GL_TEXTURE_PRIORITY 0x8066
#define GL_TEXTURE_RESIDENT 0x8067
#define GL_CLIP_PLANE0 0x3000
#define GL_CLIP_PLANE1 0x3001
#define GL_CLIP_PLANE2 0x3002
#define GL_CLIP_PLANE3 0x3003
#define GL_CLIP_PLANE4 0x3004
#define GL_CLIP_PLANE5 0x3005
#define GL_CLIP_PLANE6 0x3006
#define GL_CLIP_PLANE7 0x3007
#define GL_FOG_COORD_SRC 0x8450
#define GL_FOG_COORD 0x8451
#define GL_FRAGMENT_DEPTH 0x8452
#define GL_CURRENT_FOG_COORD 0x8453
#endif /* GL_ACCUM */

#ifndef GL_LINE_WIDTH_RANGE
#define GL_LINE_WIDTH_RANGE 0x0B22
#endif



#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifndef GL_APIENTRY
#define GL_APIENTRY
#endif

void GL_APIENTRY glNewList(GLuint list, GLenum mode);
void GL_APIENTRY glEndList();
void GL_APIENTRY glCallList(GLuint list);
void GL_APIENTRY glCallLists(GLsizei n, GLenum type, const void *lists);
void GL_APIENTRY glDeleteLists(GLuint list, GLsizei range);
GLuint GL_APIENTRY glGenLists(GLsizei range);
void GL_APIENTRY glListBase(GLuint base);
void GL_APIENTRY glBegin(GLenum mode);
void GL_APIENTRY glBitmap(GLsizei width, GLsizei height, GLfloat xorig, GLfloat yorig, GLfloat xmove, GLfloat ymove, const GLubyte *bitmap);
void GL_APIENTRY glColor3b(GLbyte red, GLbyte green, GLbyte blue);
void GL_APIENTRY glColor3bv(const GLbyte *v);
void GL_APIENTRY glColor3d(GLdouble red, GLdouble green, GLdouble blue);
void GL_APIENTRY glColor3dv(const GLdouble *v);
void GL_APIENTRY glColor3f(GLfloat red, GLfloat green, GLfloat blue);
void GL_APIENTRY glColor3fv(const GLfloat *v);
void GL_APIENTRY glColor3i(GLint red, GLint green, GLint blue);
void GL_APIENTRY glColor3iv(const GLint *v);
void GL_APIENTRY glColor3s(GLshort red, GLshort green, GLshort blue);
void GL_APIENTRY glColor3sv(const GLshort *v);
void GL_APIENTRY glColor3ub(GLubyte red, GLubyte green, GLubyte blue);
void GL_APIENTRY glColor3ubv(const GLubyte *v);
void GL_APIENTRY glColor3ui(GLuint red, GLuint green, GLuint blue);
void GL_APIENTRY glColor3uiv(const GLuint *v);
void GL_APIENTRY glColor3us(GLushort red, GLushort green, GLushort blue);
void GL_APIENTRY glColor3usv(const GLushort *v);
void GL_APIENTRY glColor4b(GLbyte red, GLbyte green, GLbyte blue, GLbyte alpha);
void GL_APIENTRY glColor4bv(const GLbyte *v);
void GL_APIENTRY glColor4d(GLdouble red, GLdouble green, GLdouble blue, GLdouble alpha);
void GL_APIENTRY glColor4dv(const GLdouble *v);
void GL_APIENTRY glColor4f(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha);
void GL_APIENTRY glColor4fv(const GLfloat *v);
void GL_APIENTRY glColor4i(GLint red, GLint green, GLint blue, GLint alpha);
void GL_APIENTRY glColor4iv(const GLint *v);
void GL_APIENTRY glColor4s(GLshort red, GLshort green, GLshort blue, GLshort alpha);
void GL_APIENTRY glColor4sv(const GLshort *v);
void GL_APIENTRY glColor4ub(GLubyte red, GLubyte green, GLubyte blue, GLubyte alpha);
void GL_APIENTRY glColor4ubv(const GLubyte *v);
void GL_APIENTRY glColor4ui(GLuint red, GLuint green, GLuint blue, GLuint alpha);
void GL_APIENTRY glColor4uiv(const GLuint *v);
void GL_APIENTRY glColor4us(GLushort red, GLushort green, GLushort blue, GLushort alpha);
void GL_APIENTRY glColor4usv(const GLushort *v);
void GL_APIENTRY glEdgeFlag(GLboolean flag);
void GL_APIENTRY glEdgeFlagv(const GLboolean *flag);
void GL_APIENTRY glEnd();
void GL_APIENTRY glIndexd(GLdouble c);
void GL_APIENTRY glIndexdv(const GLdouble *c);
void GL_APIENTRY glIndexf(GLfloat c);
void GL_APIENTRY glIndexfv(const GLfloat *c);
void GL_APIENTRY glIndexi(GLint c);
void GL_APIENTRY glIndexiv(const GLint *c);
void GL_APIENTRY glIndexs(GLshort c);
void GL_APIENTRY glIndexsv(const GLshort *c);
void GL_APIENTRY glNormal3b(GLbyte nx, GLbyte ny, GLbyte nz);
void GL_APIENTRY glNormal3bv(const GLbyte *v);
void GL_APIENTRY glNormal3d(GLdouble nx, GLdouble ny, GLdouble nz);
void GL_APIENTRY glNormal3dv(const GLdouble *v);
void GL_APIENTRY glNormal3f(GLfloat nx, GLfloat ny, GLfloat nz);
void GL_APIENTRY glNormal3fv(const GLfloat *v);
void GL_APIENTRY glNormal3i(GLint nx, GLint ny, GLint nz);
void GL_APIENTRY glNormal3iv(const GLint *v);
void GL_APIENTRY glNormal3s(GLshort nx, GLshort ny, GLshort nz);
void GL_APIENTRY glNormal3sv(const GLshort *v);
void GL_APIENTRY glRasterPos2d(GLdouble x, GLdouble y);
void GL_APIENTRY glRasterPos2dv(const GLdouble *v);
void GL_APIENTRY glRasterPos2f(GLfloat x, GLfloat y);
void GL_APIENTRY glRasterPos2fv(const GLfloat *v);
void GL_APIENTRY glRasterPos2i(GLint x, GLint y);
void GL_APIENTRY glRasterPos2iv(const GLint *v);
void GL_APIENTRY glRasterPos2s(GLshort x, GLshort y);
void GL_APIENTRY glRasterPos2sv(const GLshort *v);
void GL_APIENTRY glRasterPos3d(GLdouble x, GLdouble y, GLdouble z);
void GL_APIENTRY glRasterPos3dv(const GLdouble *v);
void GL_APIENTRY glRasterPos3f(GLfloat x, GLfloat y, GLfloat z);
void GL_APIENTRY glRasterPos3fv(const GLfloat *v);
void GL_APIENTRY glRasterPos3i(GLint x, GLint y, GLint z);
void GL_APIENTRY glRasterPos3iv(const GLint *v);
void GL_APIENTRY glRasterPos3s(GLshort x, GLshort y, GLshort z);
void GL_APIENTRY glRasterPos3sv(const GLshort *v);
void GL_APIENTRY glRasterPos4d(GLdouble x, GLdouble y, GLdouble z, GLdouble w);
void GL_APIENTRY glRasterPos4dv(const GLdouble *v);
void GL_APIENTRY glRasterPos4f(GLfloat x, GLfloat y, GLfloat z, GLfloat w);
void GL_APIENTRY glRasterPos4fv(const GLfloat *v);
void GL_APIENTRY glRasterPos4i(GLint x, GLint y, GLint z, GLint w);
void GL_APIENTRY glRasterPos4iv(const GLint *v);
void GL_APIENTRY glRasterPos4s(GLshort x, GLshort y, GLshort z, GLshort w);
void GL_APIENTRY glRasterPos4sv(const GLshort *v);
void GL_APIENTRY glRectd(GLdouble x1, GLdouble y1, GLdouble x2, GLdouble y2);
void GL_APIENTRY glRectdv(const GLdouble *v1, const GLdouble *v2);
void GL_APIENTRY glRectf(GLfloat x1, GLfloat y1, GLfloat x2, GLfloat y2);
void GL_APIENTRY glRectfv(const GLfloat *v1, const GLfloat *v2);
void GL_APIENTRY glRecti(GLint x1, GLint y1, GLint x2, GLint y2);
void GL_APIENTRY glRectiv(const GLint *v1, const GLint *v2);
void GL_APIENTRY glRects(GLshort x1, GLshort y1, GLshort x2, GLshort y2);
void GL_APIENTRY glRectsv(const GLshort *v1, const GLshort *v2);
void GL_APIENTRY glTexCoord1d(GLdouble s);
void GL_APIENTRY glTexCoord1dv(const GLdouble *v);
void GL_APIENTRY glTexCoord1f(GLfloat s);
void GL_APIENTRY glTexCoord1fv(const GLfloat *v);
void GL_APIENTRY glTexCoord1i(GLint s);
void GL_APIENTRY glTexCoord1iv(const GLint *v);
void GL_APIENTRY glTexCoord1s(GLshort s);
void GL_APIENTRY glTexCoord1sv(const GLshort *v);
void GL_APIENTRY glTexCoord2d(GLdouble s, GLdouble t);
void GL_APIENTRY glTexCoord2dv(const GLdouble *v);
void GL_APIENTRY glTexCoord2f(GLfloat s, GLfloat t);
void GL_APIENTRY glTexCoord2fv(const GLfloat *v);
void GL_APIENTRY glTexCoord2i(GLint s, GLint t);
void GL_APIENTRY glTexCoord2iv(const GLint *v);
void GL_APIENTRY glTexCoord2s(GLshort s, GLshort t);
void GL_APIENTRY glTexCoord2sv(const GLshort *v);
void GL_APIENTRY glTexCoord3d(GLdouble s, GLdouble t, GLdouble r);
void GL_APIENTRY glTexCoord3dv(const GLdouble *v);
void GL_APIENTRY glTexCoord3f(GLfloat s, GLfloat t, GLfloat r);
void GL_APIENTRY glTexCoord3fv(const GLfloat *v);
void GL_APIENTRY glTexCoord3i(GLint s, GLint t, GLint r);
void GL_APIENTRY glTexCoord3iv(const GLint *v);
void GL_APIENTRY glTexCoord3s(GLshort s, GLshort t, GLshort r);
void GL_APIENTRY glTexCoord3sv(const GLshort *v);
void GL_APIENTRY glTexCoord4d(GLdouble s, GLdouble t, GLdouble r, GLdouble q);
void GL_APIENTRY glTexCoord4dv(const GLdouble *v);
void GL_APIENTRY glTexCoord4f(GLfloat s, GLfloat t, GLfloat r, GLfloat q);
void GL_APIENTRY glTexCoord4fv(const GLfloat *v);
void GL_APIENTRY glTexCoord4i(GLint s, GLint t, GLint r, GLint q);
void GL_APIENTRY glTexCoord4iv(const GLint *v);
void GL_APIENTRY glTexCoord4s(GLshort s, GLshort t, GLshort r, GLshort q);
void GL_APIENTRY glTexCoord4sv(const GLshort *v);
void GL_APIENTRY glVertex2d(GLdouble x, GLdouble y);
void GL_APIENTRY glVertex2dv(const GLdouble *v);
void GL_APIENTRY glVertex2f(GLfloat x, GLfloat y);
void GL_APIENTRY glVertex2fv(const GLfloat *v);
void GL_APIENTRY glVertex2i(GLint x, GLint y);
void GL_APIENTRY glVertex2iv(const GLint *v);
void GL_APIENTRY glVertex2s(GLshort x, GLshort y);
void GL_APIENTRY glVertex2sv(const GLshort *v);
void GL_APIENTRY glVertex3d(GLdouble x, GLdouble y, GLdouble z);
void GL_APIENTRY glVertex3dv(const GLdouble *v);
void GL_APIENTRY glVertex3f(GLfloat x, GLfloat y, GLfloat z);
void GL_APIENTRY glVertex3fv(const GLfloat *v);
void GL_APIENTRY glVertex3i(GLint x, GLint y, GLint z);
void GL_APIENTRY glVertex3iv(const GLint *v);
void GL_APIENTRY glVertex3s(GLshort x, GLshort y, GLshort z);
void GL_APIENTRY glVertex3sv(const GLshort *v);
void GL_APIENTRY glVertex4d(GLdouble x, GLdouble y, GLdouble z, GLdouble w);
void GL_APIENTRY glVertex4dv(const GLdouble *v);
void GL_APIENTRY glVertex4f(GLfloat x, GLfloat y, GLfloat z, GLfloat w);
void GL_APIENTRY glVertex4fv(const GLfloat *v);
void GL_APIENTRY glVertex4i(GLint x, GLint y, GLint z, GLint w);
void GL_APIENTRY glVertex4iv(const GLint *v);
void GL_APIENTRY glVertex4s(GLshort x, GLshort y, GLshort z, GLshort w);
void GL_APIENTRY glVertex4sv(const GLshort *v);
void GL_APIENTRY glClipPlane(GLenum plane, const GLdouble *equation);
void GL_APIENTRY glColorMaterial(GLenum face, GLenum mode);
void GL_APIENTRY glFogf(GLenum pname, GLfloat param);
void GL_APIENTRY glFogfv(GLenum pname, const GLfloat *params);
void GL_APIENTRY glFogi(GLenum pname, GLint param);
void GL_APIENTRY glFogiv(GLenum pname, const GLint *params);
void GL_APIENTRY glLightf(GLenum light, GLenum pname, GLfloat param);
void GL_APIENTRY glLightfv(GLenum light, GLenum pname, const GLfloat *params);
void GL_APIENTRY glLighti(GLenum light, GLenum pname, GLint param);
void GL_APIENTRY glLightiv(GLenum light, GLenum pname, const GLint *params);
void GL_APIENTRY glLightModelf(GLenum pname, GLfloat param);
void GL_APIENTRY glLightModelfv(GLenum pname, const GLfloat *params);
void GL_APIENTRY glLightModeli(GLenum pname, GLint param);
void GL_APIENTRY glLightModeliv(GLenum pname, const GLint *params);
void GL_APIENTRY glLineStipple(GLint factor, GLushort pattern);
void GL_APIENTRY glMaterialf(GLenum face, GLenum pname, GLfloat param);
void GL_APIENTRY glMaterialfv(GLenum face, GLenum pname, const GLfloat *params);
void GL_APIENTRY glMateriali(GLenum face, GLenum pname, GLint param);
void GL_APIENTRY glMaterialiv(GLenum face, GLenum pname, const GLint *params);
void GL_APIENTRY glPolygonStipple(const GLubyte *mask);
void GL_APIENTRY glShadeModel(GLenum mode);
void GL_APIENTRY glTexEnvf(GLenum target, GLenum pname, GLfloat param);
void GL_APIENTRY glTexEnvfv(GLenum target, GLenum pname, const GLfloat *params);
void GL_APIENTRY glTexEnvi(GLenum target, GLenum pname, GLint param);
void GL_APIENTRY glTexEnviv(GLenum target, GLenum pname, const GLint *params);
void GL_APIENTRY glTexGend(GLenum coord, GLenum pname, GLdouble param);
void GL_APIENTRY glTexGendv(GLenum coord, GLenum pname, const GLdouble *params);
void GL_APIENTRY glTexGenf(GLenum coord, GLenum pname, GLfloat param);
void GL_APIENTRY glTexGenfv(GLenum coord, GLenum pname, const GLfloat *params);
void GL_APIENTRY glTexGeni(GLenum coord, GLenum pname, GLint param);
void GL_APIENTRY glTexGeniv(GLenum coord, GLenum pname, const GLint *params);
void GL_APIENTRY glFeedbackBuffer(GLsizei size, GLenum type, GLfloat *buffer);
void GL_APIENTRY glSelectBuffer(GLsizei size, GLuint *buffer);
GLint GL_APIENTRY glRenderMode(GLenum mode);
void GL_APIENTRY glInitNames();
void GL_APIENTRY glLoadName(GLuint name);
void GL_APIENTRY glPassThrough(GLfloat token);
void GL_APIENTRY glPopName();
void GL_APIENTRY glPushName(GLuint name);
void GL_APIENTRY glClearAccum(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha);
void GL_APIENTRY glClearIndex(GLfloat c);
void GL_APIENTRY glIndexMask(GLuint mask);
void GL_APIENTRY glAccum(GLenum op, GLfloat value);
void GL_APIENTRY glPopAttrib();
void GL_APIENTRY glPushAttrib(GLbitfield mask);
void GL_APIENTRY glMap1d(GLenum target, GLdouble u1, GLdouble u2, GLint stride, GLint order, const GLdouble *points);
void GL_APIENTRY glMap1f(GLenum target, GLfloat u1, GLfloat u2, GLint stride, GLint order, const GLfloat *points);
void GL_APIENTRY glMap2d(GLenum target, GLdouble u1, GLdouble u2, GLint ustride, GLint uorder, GLdouble v1, GLdouble v2, GLint vstride, GLint vorder, const GLdouble *points);
void GL_APIENTRY glMap2f(GLenum target, GLfloat u1, GLfloat u2, GLint ustride, GLint uorder, GLfloat v1, GLfloat v2, GLint vstride, GLint vorder, const GLfloat *points);
void GL_APIENTRY glMapGrid1d(GLint un, GLdouble u1, GLdouble u2);
void GL_APIENTRY glMapGrid1f(GLint un, GLfloat u1, GLfloat u2);
void GL_APIENTRY glMapGrid2d(GLint un, GLdouble u1, GLdouble u2, GLint vn, GLdouble v1, GLdouble v2);
void GL_APIENTRY glMapGrid2f(GLint un, GLfloat u1, GLfloat u2, GLint vn, GLfloat v1, GLfloat v2);
void GL_APIENTRY glEvalCoord1d(GLdouble u);
void GL_APIENTRY glEvalCoord1dv(const GLdouble *u);
void GL_APIENTRY glEvalCoord1f(GLfloat u);
void GL_APIENTRY glEvalCoord1fv(const GLfloat *u);
void GL_APIENTRY glEvalCoord2d(GLdouble u, GLdouble v);
void GL_APIENTRY glEvalCoord2dv(const GLdouble *u);
void GL_APIENTRY glEvalCoord2f(GLfloat u, GLfloat v);
void GL_APIENTRY glEvalCoord2fv(const GLfloat *u);
void GL_APIENTRY glEvalMesh1(GLenum mode, GLint i1, GLint i2);
void GL_APIENTRY glEvalPoint1(GLint i);
void GL_APIENTRY glEvalMesh2(GLenum mode, GLint i1, GLint i2, GLint j1, GLint j2);
void GL_APIENTRY glEvalPoint2(GLint i, GLint j);
void GL_APIENTRY glAlphaFunc(GLenum func, GLfloat ref);
void GL_APIENTRY glPixelZoom(GLfloat xfactor, GLfloat yfactor);
void GL_APIENTRY glPixelTransferf(GLenum pname, GLfloat param);
void GL_APIENTRY glPixelTransferi(GLenum pname, GLint param);
void GL_APIENTRY glPixelMapfv(GLenum map, GLsizei mapsize, const GLfloat *values);
void GL_APIENTRY glPixelMapuiv(GLenum map, GLsizei mapsize, const GLuint *values);
void GL_APIENTRY glPixelMapusv(GLenum map, GLsizei mapsize, const GLushort *values);
void GL_APIENTRY glCopyPixels(GLint x, GLint y, GLsizei width, GLsizei height, GLenum type);
void GL_APIENTRY glDrawPixels(GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels);
void GL_APIENTRY glGetClipPlane(GLenum plane, GLdouble *equation);
void GL_APIENTRY glGetLightfv(GLenum light, GLenum pname, GLfloat *params);
void GL_APIENTRY glGetLightiv(GLenum light, GLenum pname, GLint *params);
void GL_APIENTRY glGetMapdv(GLenum target, GLenum query, GLdouble *v);
void GL_APIENTRY glGetMapfv(GLenum target, GLenum query, GLfloat *v);
void GL_APIENTRY glGetMapiv(GLenum target, GLenum query, GLint *v);
void GL_APIENTRY glGetMaterialfv(GLenum face, GLenum pname, GLfloat *params);
void GL_APIENTRY glGetMaterialiv(GLenum face, GLenum pname, GLint *params);
void GL_APIENTRY glGetPixelMapfv(GLenum map, GLfloat *values);
void GL_APIENTRY glGetPixelMapuiv(GLenum map, GLuint *values);
void GL_APIENTRY glGetPixelMapusv(GLenum map, GLushort *values);
void GL_APIENTRY glGetPolygonStipple(GLubyte *mask);
void GL_APIENTRY glGetTexEnvfv(GLenum target, GLenum pname, GLfloat *params);
void GL_APIENTRY glGetTexEnviv(GLenum target, GLenum pname, GLint *params);
void GL_APIENTRY glGetTexGendv(GLenum coord, GLenum pname, GLdouble *params);
void GL_APIENTRY glGetTexGenfv(GLenum coord, GLenum pname, GLfloat *params);
void GL_APIENTRY glGetTexGeniv(GLenum coord, GLenum pname, GLint *params);
GLboolean GL_APIENTRY glIsList(GLuint list);
void GL_APIENTRY glFrustum(GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble zNear, GLdouble zFar);
void GL_APIENTRY glLoadIdentity();
void GL_APIENTRY glLoadMatrixf(const GLfloat *m);
void GL_APIENTRY glLoadMatrixd(const GLdouble *m);
void GL_APIENTRY glMatrixMode(GLenum mode);
void GL_APIENTRY glMultMatrixf(const GLfloat *m);
void GL_APIENTRY glMultMatrixd(const GLdouble *m);
void GL_APIENTRY glOrtho(GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble zNear, GLdouble zFar);
void GL_APIENTRY glPopMatrix();
void GL_APIENTRY glPushMatrix();
void GL_APIENTRY glRotated(GLdouble angle, GLdouble x, GLdouble y, GLdouble z);
void GL_APIENTRY glRotatef(GLfloat angle, GLfloat x, GLfloat y, GLfloat z);
void GL_APIENTRY glScaled(GLdouble x, GLdouble y, GLdouble z);
void GL_APIENTRY glScalef(GLfloat x, GLfloat y, GLfloat z);
void GL_APIENTRY glTranslated(GLdouble x, GLdouble y, GLdouble z);
void GL_APIENTRY glTranslatef(GLfloat x, GLfloat y, GLfloat z);
void GL_APIENTRY glArrayElement(GLint i);
void GL_APIENTRY glColorPointer(GLint size, GLenum type, GLsizei stride, const void *pointer);
void GL_APIENTRY glEdgeFlagPointer(GLsizei stride, const void *pointer);
void GL_APIENTRY glIndexPointer(GLenum type, GLsizei stride, const void *pointer);
void GL_APIENTRY glInterleavedArrays(GLenum format, GLsizei stride, const void *pointer);
void GL_APIENTRY glNormalPointer(GLenum type, GLsizei stride, const void *pointer);
void GL_APIENTRY glTexCoordPointer(GLint size, GLenum type, GLsizei stride, const void *pointer);
void GL_APIENTRY glVertexPointer(GLint size, GLenum type, GLsizei stride, const void *pointer);
GLboolean GL_APIENTRY glAreTexturesResident(GLsizei n, const GLuint *textures, GLboolean *residences);
void GL_APIENTRY glPrioritizeTextures(GLsizei n, const GLuint *textures, const GLfloat *priorities);
void GL_APIENTRY glPopClientAttrib();
void GL_APIENTRY glPushClientAttrib(GLbitfield mask);
void GL_APIENTRY glClientActiveTexture(GLenum texture);
void GL_APIENTRY glMultiTexCoord1d(GLenum target, GLdouble s);
void GL_APIENTRY glMultiTexCoord1dv(GLenum target, const GLdouble *v);
void GL_APIENTRY glMultiTexCoord1f(GLenum target, GLfloat s);
void GL_APIENTRY glMultiTexCoord1fv(GLenum target, const GLfloat *v);
void GL_APIENTRY glMultiTexCoord1i(GLenum target, GLint s);
void GL_APIENTRY glMultiTexCoord1iv(GLenum target, const GLint *v);
void GL_APIENTRY glMultiTexCoord1s(GLenum target, GLshort s);
void GL_APIENTRY glMultiTexCoord1sv(GLenum target, const GLshort *v);
void GL_APIENTRY glMultiTexCoord2d(GLenum target, GLdouble s, GLdouble t);
void GL_APIENTRY glMultiTexCoord2dv(GLenum target, const GLdouble *v);
void GL_APIENTRY glMultiTexCoord2f(GLenum target, GLfloat s, GLfloat t);
void GL_APIENTRY glMultiTexCoord2fv(GLenum target, const GLfloat *v);
void GL_APIENTRY glMultiTexCoord2i(GLenum target, GLint s, GLint t);
void GL_APIENTRY glMultiTexCoord2iv(GLenum target, const GLint *v);
void GL_APIENTRY glMultiTexCoord2s(GLenum target, GLshort s, GLshort t);
void GL_APIENTRY glMultiTexCoord2sv(GLenum target, const GLshort *v);
void GL_APIENTRY glMultiTexCoord3d(GLenum target, GLdouble s, GLdouble t, GLdouble r);
void GL_APIENTRY glMultiTexCoord3dv(GLenum target, const GLdouble *v);
void GL_APIENTRY glMultiTexCoord3f(GLenum target, GLfloat s, GLfloat t, GLfloat r);
void GL_APIENTRY glMultiTexCoord3fv(GLenum target, const GLfloat *v);
void GL_APIENTRY glMultiTexCoord3i(GLenum target, GLint s, GLint t, GLint r);
void GL_APIENTRY glMultiTexCoord3iv(GLenum target, const GLint *v);
void GL_APIENTRY glMultiTexCoord3s(GLenum target, GLshort s, GLshort t, GLshort r);
void GL_APIENTRY glMultiTexCoord3sv(GLenum target, const GLshort *v);
void GL_APIENTRY glMultiTexCoord4d(GLenum target, GLdouble s, GLdouble t, GLdouble r, GLdouble q);
void GL_APIENTRY glMultiTexCoord4dv(GLenum target, const GLdouble *v);
void GL_APIENTRY glMultiTexCoord4f(GLenum target, GLfloat s, GLfloat t, GLfloat r, GLfloat q);
void GL_APIENTRY glMultiTexCoord4fv(GLenum target, const GLfloat *v);
void GL_APIENTRY glMultiTexCoord4i(GLenum target, GLint s, GLint t, GLint r, GLint q);
void GL_APIENTRY glMultiTexCoord4iv(GLenum target, const GLint *v);
void GL_APIENTRY glMultiTexCoord4s(GLenum target, GLshort s, GLshort t, GLshort r, GLshort q);
void GL_APIENTRY glMultiTexCoord4sv(GLenum target, const GLshort *v);
void GL_APIENTRY glLoadTransposeMatrixf(const GLfloat *m);
void GL_APIENTRY glLoadTransposeMatrixd(const GLdouble *m);
void GL_APIENTRY glMultTransposeMatrixf(const GLfloat *m);
void GL_APIENTRY glMultTransposeMatrixd(const GLdouble *m);
void GL_APIENTRY glFogCoordf(GLfloat coord);
void GL_APIENTRY glFogCoordfv(const GLfloat *coord);
void GL_APIENTRY glFogCoordd(GLdouble coord);
void GL_APIENTRY glFogCoorddv(const GLdouble *coord);
void GL_APIENTRY glFogCoordPointer(GLenum type, GLsizei stride, const void *pointer);
void GL_APIENTRY glSecondaryColor3b(GLbyte red, GLbyte green, GLbyte blue);
void GL_APIENTRY glSecondaryColor3bv(const GLbyte *v);
void GL_APIENTRY glSecondaryColor3d(GLdouble red, GLdouble green, GLdouble blue);
void GL_APIENTRY glSecondaryColor3dv(const GLdouble *v);
void GL_APIENTRY glSecondaryColor3f(GLfloat red, GLfloat green, GLfloat blue);
void GL_APIENTRY glSecondaryColor3fv(const GLfloat *v);
void GL_APIENTRY glSecondaryColor3i(GLint red, GLint green, GLint blue);
void GL_APIENTRY glSecondaryColor3iv(const GLint *v);
void GL_APIENTRY glSecondaryColor3s(GLshort red, GLshort green, GLshort blue);
void GL_APIENTRY glSecondaryColor3sv(const GLshort *v);
void GL_APIENTRY glSecondaryColor3ub(GLubyte red, GLubyte green, GLubyte blue);
void GL_APIENTRY glSecondaryColor3ubv(const GLubyte *v);
void GL_APIENTRY glSecondaryColor3ui(GLuint red, GLuint green, GLuint blue);
void GL_APIENTRY glSecondaryColor3uiv(const GLuint *v);
void GL_APIENTRY glSecondaryColor3us(GLushort red, GLushort green, GLushort blue);
void GL_APIENTRY glSecondaryColor3usv(const GLushort *v);
void GL_APIENTRY glSecondaryColorPointer(GLint size, GLenum type, GLsizei stride, const void *pointer);
void GL_APIENTRY glWindowPos2d(GLdouble x, GLdouble y);
void GL_APIENTRY glWindowPos2dv(const GLdouble *v);
void GL_APIENTRY glWindowPos2f(GLfloat x, GLfloat y);
void GL_APIENTRY glWindowPos2fv(const GLfloat *v);
void GL_APIENTRY glWindowPos2i(GLint x, GLint y);
void GL_APIENTRY glWindowPos2iv(const GLint *v);
void GL_APIENTRY glWindowPos2s(GLshort x, GLshort y);
void GL_APIENTRY glWindowPos2sv(const GLshort *v);
void GL_APIENTRY glWindowPos3d(GLdouble x, GLdouble y, GLdouble z);
void GL_APIENTRY glWindowPos3dv(const GLdouble *v);
void GL_APIENTRY glWindowPos3f(GLfloat x, GLfloat y, GLfloat z);
void GL_APIENTRY glWindowPos3fv(const GLfloat *v);
void GL_APIENTRY glWindowPos3i(GLint x, GLint y, GLint z);
void GL_APIENTRY glWindowPos3iv(const GLint *v);
void GL_APIENTRY glWindowPos3s(GLshort x, GLshort y, GLshort z);
void GL_APIENTRY glWindowPos3sv(const GLshort *v);
void GL_APIENTRY glVertexP2ui(GLenum type, GLuint value);
void GL_APIENTRY glVertexP2uiv(GLenum type, const GLuint *value);
void GL_APIENTRY glVertexP3ui(GLenum type, GLuint value);
void GL_APIENTRY glVertexP3uiv(GLenum type, const GLuint *value);
void GL_APIENTRY glVertexP4ui(GLenum type, GLuint value);
void GL_APIENTRY glVertexP4uiv(GLenum type, const GLuint *value);
void GL_APIENTRY glColorP3ui(GLenum type, GLuint color);
void GL_APIENTRY glColorP3uiv(GLenum type, const GLuint *color);
void GL_APIENTRY glColorP4ui(GLenum type, GLuint color);
void GL_APIENTRY glColorP4uiv(GLenum type, const GLuint *color);
void GL_APIENTRY glGetnPolygonStipple(GLsizei bufSize, GLubyte *pattern);

#ifdef __cplusplus
}
#endif

#endif /* GL_COMPAT_H */