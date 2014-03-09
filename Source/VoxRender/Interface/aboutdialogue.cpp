/* ===========================================================================

	Project: VoxRender - About Dialogue

	Description: Implements the about dialogue containing program info

    Copyright (C) 2012 Lucas Sherman

	Lucas Sherman, email: LucasASherman@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

=========================================================================== */

// Include Headers
#include "aboutdialogue.h"
#include "ui_aboutdialogue.h"

// Include Dependencies
#include "VoxLib/Core/Format.h"
#include "VoxLib/Core/Version.h"

// Qt4 Includes
#include <QTextBlockFormat>
#include <QTextCursor>

// ---------------------------------------------------------
// Constructor - Initialize the about box image
// ---------------------------------------------------------
AboutDialogue::AboutDialogue( QWidget *parent ) :
    QDialog(parent),
    ui(new Ui::AboutDialogue),
    m_imageView(this)
{
    ui->setupUi(this);
     
	connect(&m_imageView, SIGNAL(clicked()), this, SLOT(close()));
}

// ---------------------------------------------------------
//  Destructor
// ---------------------------------------------------------
AboutDialogue::~AboutDialogue( )
{
    delete ui;
}

// ---------------------------------------------------------
//  Constructor - Initializes the image structures
// ---------------------------------------------------------
AboutImage::AboutImage( QWidget *parent ) : 
    QGraphicsView(parent),
    m_authors("VoxRender is an open source development project.    "
              "Visit the website at http://code.google.com/p/voxrender/.    "    
              "VoxRender uses open source software: libpng, libjpeg, cURL, Boost C++ Libraries, QT5, ExposureRender, LuxRender"
              ),
    m_version(vox::format("Version %1%", VOX_VERSION_STRING).c_str())
{
	setBackgroundBrush(QImage(":/images/splash.png"));
	setCacheMode(QGraphicsView::CacheBackground);
	
	m_authors.setDefaultTextColor(Qt::white);
	m_authors.setPos(540, 288);
    
    m_version.setDefaultTextColor(Qt::white);
    auto font = m_version.font();
    font.setPixelSize(18);
    m_version.setFont(font);

    QFontMetrics metrics(font);
    int width = metrics.width(m_version.toPlainText());
    m_version.setPos(275-width/2, 250);

	m_scene.setSceneRect(0, 0, 550, 330);
	m_scene.addItem(&m_authors);
    m_scene.addItem(&m_version);
	setScene(&m_scene);

    // Text scrolling update 
	m_scrolltimer.start(10);
	connect(&m_scrolltimer, SIGNAL(timeout()), SLOT(scrollTimeout()));
}

// ---------------------------------------------------------
//  Scrolls the info text across the about image
// ---------------------------------------------------------
void AboutImage::scrollTimeout()
{
	auto xpos = m_authors.x();
	auto endpos = xpos + m_authors.sceneBoundingRect().width();

	if (endpos < 0) xpos = 540.0f;
	else xpos = xpos - 1.0f;
	
	m_authors.setPos(xpos, m_authors.y());
}

// ---------------------------------------------------------
//  Signal so the parent dialogue can complete execution
// ---------------------------------------------------------
void AboutImage::mousePressEvent( QMouseEvent* event ) 
{
	emit clicked();
}
