module;

#include "macros.hpp"
#include "primitive_types.hpp"
#include <ft2build.h>
#include FT_FREETYPE_H

export module vulkan_app:text;

import vulkan_hpp;
import std;

// This structure holds all the information needed to render a single character.
export struct GlyphInfo {
  // Texture coordinates for the glyph in the atlas
  float uv_x0;
  float uv_y0;
  float uv_x1;
  float uv_y1;

  // Size of the glyph quad in pixels
  float width;
  float height;

  // The offset from the text cursor's baseline to the glyph's top-left corner
  float bearing_x;
  float bearing_y;

  // The horizontal distance to advance the cursor to the next character
  float advance;
};

// This struct holds the complete output of the font atlas generation.
export struct FontAtlasData {
  std::vector<unsigned char> atlasBitmap;
  int atlasWidth;
  int atlasHeight;
  std::map<char, GlyphInfo> glyphs;
};

/**
 * @brief Generates a font atlas for the specified font file and pixel height.
 * This function handles loading a TTF font file, calculating an optimal size for
 * a texture atlas, rendering the glyphs for a specified character set (ASCII 32-126)
 * into the atlas, and packaging all relevant data into a single struct.
 *
 * @param fontPath Path to the .ttf font file.
 * @param pixelHeight The desired height of the font in pixels.
 * @return A FontAtlasData struct containing the bitmap, dimensions, and glyph metrics.
 */
export [[nodiscard]] std::expected<FontAtlasData, std::string>
createFontAtlas(const std::string &fontPath, int pixelHeight) {
  FontAtlasData atlasData;

  FT_Library ft;
  if (FT_Init_FreeType(&ft)) {
    return std::unexpected("FREETYPE: Could not init FreeType Library");
  }

  FT_Face face;
  if (FT_New_Face(ft, fontPath.c_str(), 0, &face)) {
    FT_Done_FreeType(ft);
    return std::unexpected("FREETYPE: Failed to load font: " + fontPath);
  }

  FT_Set_Pixel_Sizes(face, 0, pixelHeight);

  // Disable byte-alignment restriction for pixel unpacking
  // This is important for tightly packed glyphs in the atlas.
  // glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  int atlasWidth = 0;
  int atlasHeight = 0;
  int rowHeight = 0;
  int penX = 0;

  // --- Pass 1: Calculate required atlas dimensions ---
  // A more robust approach than summing area is to simulate the packing.
  for (unsigned char c = 32; c < 127; ++c) {
    if (FT_Load_Char(face, c, FT_LOAD_RENDER)) {
      std::println("Warning: Failed to load Glyph for character '{}'", c);
      continue;
    }
    if (penX + face->glyph->bitmap.width + 1 >=
        512) { // Use a fixed-width for simplicity (e.g. 512px)
      atlasWidth = std::max(atlasWidth, penX);
      penX = 0;
      atlasHeight += rowHeight;
      rowHeight = 0;
    }
    penX += face->glyph->bitmap.width + 1;
    rowHeight = std::max(rowHeight, (int)face->glyph->bitmap.rows);
  }
  atlasWidth = std::max(atlasWidth, penX);
  atlasHeight += rowHeight;

  atlasData.atlasWidth = atlasWidth;
  atlasData.atlasHeight = atlasHeight;
  atlasData.atlasBitmap.resize(atlasData.atlasWidth * atlasData.atlasHeight, 0);

  // --- Pass 2: Pack glyphs and generate the atlas ---
  penX = 0;
  int penY = 0;
  rowHeight = 0;

  for (unsigned char c = 32; c < 127; ++c) {
    if (FT_Load_Char(face, c, FT_LOAD_RENDER)) {
      continue;
    }

    FT_GlyphSlot glyph = face->glyph;

    if (penX + glyph->bitmap.width + 1 >= atlasData.atlasWidth) {
      penY += rowHeight;
      penX = 0;
      rowHeight = 0;
    }

    // Copy glyph bitmap to our main atlas bitmap
    for (unsigned int y = 0; y < glyph->bitmap.rows; ++y) {
      for (unsigned int x = 0; x < glyph->bitmap.width; ++x) {
        int atlasIndex = (penY + y) * atlasData.atlasWidth + (penX + x);
        int glyphIndex = y * glyph->bitmap.pitch + x;
        atlasData.atlasBitmap[atlasIndex] = glyph->bitmap.buffer[glyphIndex];
      }
    }

    // Store glyph metrics. UVs are calculated based on the final, known atlas dimensions.
    GlyphInfo info{
        .uv_x0 = static_cast<float>(penX) / atlasData.atlasWidth,
        .uv_y0 = static_cast<float>(penY) / atlasData.atlasHeight,
        .uv_x1 = static_cast<float>(penX + glyph->bitmap.width) / atlasData.atlasWidth,
        .uv_y1 = static_cast<float>(penY + glyph->bitmap.rows) / atlasData.atlasHeight,
        .width = static_cast<float>(glyph->bitmap.width),
        .height = static_cast<float>(glyph->bitmap.rows),
        .bearing_x = static_cast<float>(glyph->metrics.horiBearingX >> 6),
        .bearing_y = static_cast<float>(glyph->metrics.horiBearingY >> 6),
        .advance = static_cast<float>(glyph->metrics.horiAdvance >> 6),
    };

    atlasData.glyphs[c] = info;

    penX += glyph->bitmap.width + 1;
    rowHeight = std::max(rowHeight, (int)glyph->bitmap.rows);
  }

  // Create a fallback '?' glyph for any character not in the atlas
  if (atlasData.glyphs.find('?') == atlasData.glyphs.end()) {
    if (atlasData.glyphs.count('A')) {
      atlasData.glyphs['?'] = atlasData.glyphs['A'];
    } else if (!atlasData.glyphs.empty()) {
      atlasData.glyphs['?'] = atlasData.glyphs.begin()->second;
    }
  }

  FT_Done_Face(face);
  FT_Done_FreeType(ft);

  return atlasData;
}
