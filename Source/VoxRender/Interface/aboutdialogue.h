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

// Begin definition
#ifndef ABOUTDIALOGUE_H
#define ABOUTDIALOGUE_H

// Qt4 Includes
#include <QtCore/QTimer>
#include <QtGui/QImage>
#include <QtWidgets/QDialog>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QGraphicsScene>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGraphicsTextItem>

namespace Ui { class AboutDialogue; }

// Dialogue graphics view
class AboutImage : public QGraphicsView
{
	Q_OBJECT

public:
	AboutImage( QWidget *parent = 0 );

protected:
	void mousePressEvent( QMouseEvent* event );

private:
	QGraphicsScene m_scene;
	QGraphicsTextItem m_authors;
	QGraphicsTextItem m_version;
	QTimer m_scrolltimer;

private slots:
	void scrollTimeout( );

signals:
	void clicked( );

};

// VoxRender about dialogue
class AboutDialogue : public QDialog
{
    Q_OBJECT
    
public:
    explicit AboutDialogue( QWidget *parent = 0 );
    ~AboutDialogue( );
    
private:
    Ui::AboutDialogue *ui;

    AboutImage m_imageView;
};

// End definition
#endif // ABOUTDIALOGUE_H
