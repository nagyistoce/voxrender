/* ===========================================================================

    Project: VoxRender
    
	Description: Implements an accordion style pane widgeth

    Copyright (C) 2013 Lucas Sherman

    MODIFIED FROM LUXRENDER'S 'panewidget.cpp'

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
#ifndef PANEWIDGET_H
#define PANEWIDGET_H

#include <QtWidgets/QWidget>
#include <QtGui/QPixmap>
#include <QtWidgets/QLabel>

namespace Ui
{
	class PaneWidget;
	class ClickableLabel;
}

class ClickableLabel : public QLabel
{
    Q_OBJECT

public:

	ClickableLabel( const QString& label = "", QWidget *parent = 0 );
	~ClickableLabel( ) {};

protected:

	void mouseReleaseEvent( QMouseEvent* event );

signals:

	void clicked( );

};

enum SoloState
{
	SOLO_OFF,
	SOLO_ON,
	SOLO_ENABLED
};

class PaneWidget : public QWidget
{
	Q_OBJECT

public:

	PaneWidget( QWidget *parent, const QString& label = "", 
		const QString& icon = "", bool onoffbutton=false, 
		bool solobutton=false );

    /** Sets the title which appears in the bar of the pane */
	void setTitle(QString const& title);

    /** Returns the title of the pane */
    QString const& title();

    /** Sets the icon in the upper left corner of the pane */
	void setIcon(QString const& icon);

	void setWidget( QWidget *widget );
	QWidget *getWidget( );

	void showOnOffButton( bool showbutton = true );
	void showSoloButton( bool showbutton = true );
	void expand();
	void collapse();

	bool powerON;
	
	SoloState m_SoloState;
	void SetSolo( SoloState );

	int m_Index;

	inline int  GetIndex( ) { return m_Index; }
	inline void SetIndex( int Index ) { m_Index = Index; }

private:

	Ui::PaneWidget *ui;

	QWidget *mainwidget;
	QPixmap expandedicon, collapsedicon;
	std::unique_ptr<ClickableLabel> expandlabel;
	std::unique_ptr<ClickableLabel> onofflabel;
	std::unique_ptr<ClickableLabel> sololabel;

	bool expanded;

signals:

	void valuesChanged( );

	void turnedOn();
	void turnedOff();

	void signalLightGroupSolo( int index );

private slots:

	void expandClicked( );
	void onoffClicked( );
	void soloClicked( );
  
};

// End definition
#endif // PANEWIDGET_H

