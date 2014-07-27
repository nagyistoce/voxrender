/* ===========================================================================

	Project: TransferWidget - Transfer Function Widget

	Description:
	 Implements an interface for interactive transfer function modification.

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

// Begin Definition
#ifndef TRANSFERWIDGET_H
#define TRANSFERWIDGET_H

// Include Dependencies
#include "histogramview.h"
#include "Extensions/QColorPushButton.h"

// VoxLib Includes
#include "VoxScene/Transfer.h"
#include "VoxScene/Scene.h"

// QT Includes
#include <QtWidgets/QWidget>
#include <QtWidgets/QColorDialog>

namespace Ui { class TransferWidget; }

// Transfer modification widget
class TransferWidget : public QWidget
{
    Q_OBJECT
    
public:
    /** Destructor */
    ~TransferWidget();

    /** Instance method for transfer function widget */
    static TransferWidget * instance()
    {
        static TransferWidget * singleton = nullptr;
        if (!singleton) 
        {
            singleton = new TransferWidget();
        }
        return singleton;
    }

    /** Loads a new transfer function */
    void load();

    /** Saves the transfer function */
    void save();
    
protected:
    friend TransferItem;

    /** Node change event type */
    enum EditType
    {
        EditType_Begin,
        EditType_Attr = EditType_Begin, ///< Node attribute changed
        EditType_Reference,             ///< Current node changed
        EditType_Create,                ///< Node was created
        EditType_Remove,                ///< Node was removed
        EditType_End,
    };

    /** Validates a node position change event and returns the resolved position */
    void validateNodePositionChange(float * x, float * y = nullptr, float * z = nullptr);

    /** Returns the currently selected transfer function node */
    std::shared_ptr<vox::Node> selectedNode();
    
    /** Returns the currently selected transfer function node */
    std::shared_ptr<vox::Quad> selectedQuad(vox::Quad::Node * node = nullptr);

    /** Sets the active 1D transfer function node for the widget */
    void setSelectedNode(std::shared_ptr<vox::Node> node, bool signal = true);

    /** Sets the active 2D transfer function node for the widget */
    void setSelectedQuad(std::shared_ptr<vox::Quad> quad, vox::Quad::Node node = vox::Quad::Node_End, bool signal = true);
    
    /** Notifies the widget of a node position change */
    void onNodeChanged();

private:
    explicit TransferWidget(QWidget *parent = 0);

    Ui::TransferWidget * ui;

    bool m_ignore;

	// Color Selection Dialogue
    QColorPushButton * m_colorDiffuse;
    QColorPushButton * m_colorEmissive;
    QColorPushButton * m_colorSpecular;

	// Transfer function view
	HistogramView * m_primaryView;
	HistogramView * m_secondaryView;
	int             m_dimensions;

	// Transfer function information
    std::shared_ptr<vox::Transfer> m_transfer;
    std::shared_ptr<vox::Node>     m_currentNode;
    std::shared_ptr<vox::Quad>     m_currentQuad;
    std::shared_ptr<vox::Material> m_currentMaterial;
    vox::Quad::Node m_currentIndex;

    // Edit clones for managing undo/redo operations
    std::shared_ptr<vox::Material> m_editMaterial;

	// Dimension selection
	void switchDimensions(int nDims);
	bool canSwitchDimensions();
    
    void keyPressEvent(QKeyEvent * event);

    /** Sets the active material for editing in the widget */
    void setSelectedMaterial(std::shared_ptr<vox::Material> material);

    /** Updates the transfer function map's resolution */
    void updateResolution();

signals:
    /** Issued when a different node has been set as the active node */
    void transferNodeSelected(std::shared_ptr<vox::Node> node); 

    /** Issued when a different quad has been set as the active quad */
    void transferQuadSelected(std::shared_ptr<vox::Quad> quad, vox::Quad::Node node); 
    
    /** Issued when the current transfer function node/etc. has been modified */
    void nodeChanged(int editType = EditType_Attr);

private slots:
    void beginMaterialChange();
    void endMaterialChange();
    void sceneChanged(vox::Scene & scene, void * userInfo);

	// Node selection group box
	void on_pushButton_delete_clicked();
	void on_pushButton_next_clicked();
	void on_pushButton_prev_clicked();
	void on_pushButton_first_clicked();
	void on_pushButton_last_clicked();

	// Dimension selection group box
	void on_radioButton_1_toggled(bool checked);
	void on_radioButton_2_toggled(bool checked);
	void on_radioButton_3_toggled(bool checked);

    // Slider/SpinBox connections
    void on_doubleSpinBox_opacity_valueChanged(double value);
    void on_doubleSpinBox_density_valueChanged(double value);
    void on_doubleSpinBox_gloss_valueChanged(double value);
    void on_doubleSpinBox_emissiveStr_valueChanged(double value);
    void on_horizontalSlider_opacity_valueChanged(int value);
    void on_horizontalSlider_density_valueChanged(int value);
    void on_horizontalSlider_gloss_valueChanged(int value);
    void on_horizontalSlider_emissiveStr_valueChanged(int value);
    
    void on_spinBox_resX_valueChanged(int value);
    void on_spinBox_resY_valueChanged(int value);
    void on_spinBox_resZ_valueChanged(int value);

    // Color selection widgets
    void colorDiffuseChanged(QColor const& color);
    void colorEmissiveChanged(QColor const& color);
    void colorSpecularChanged(QColor const& color);
};

#endif // TRANSFERWIDGET_H
