using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using UnityEngine;
using UnityEngine.UI;

public class CameraTracker : MonoBehaviour
{
	private float refreshDisplayFreq = 1;

	private float displayDelta = 0;

    public GameObject target;
    public float height;
    public float distance;

	private bool TopView = true;

	public Text displayTimeToLeft;

	public Text gameStatus;

	public Button btnChangeView;

	private static Vector3 cameraPosition;
	private static Quaternion cameraRotation;


	private static bool firstEpisode = true;

    // Start is called before the first frame update
    void Start()
    {
        displayDelta = 0;

		btnChangeView.onClick.AddListener(TaskOnClick);

		refreshDisplayFreq = 1;
		if (firstEpisode) {
			cameraPosition = gameObject.transform.position;
			cameraRotation = gameObject.transform.rotation;
			firstEpisode = false;
			GetComponent<Camera>().orthographic = true;
		} else {
			gameObject.transform.position = cameraPosition;
			gameObject.transform.rotation = cameraRotation;
		}
	}

	public void TaskOnClick()
    {
		TopView = !TopView;
		if (TopView)
        {
			transform.position = cameraPosition;
			transform.rotation = cameraRotation;
			GetComponent<Camera>().orthographic = true;
		}
	}

	/// <summary>
	/// This function is called every fixed framerate frame, if the MonoBehaviour is enabled.
	/// </summary>
	void FixedUpdate()
	{
		
	}

	void Update()
    {
		
		if (target != null) {
			if (!TopView)
            {
				transform.position = target.transform.position - target.transform.forward * distance;
				GetComponent<Camera>().orthographic = false;
				Vector3 pos = gameObject.transform.position;
				gameObject.transform.position = new Vector3(pos.x, pos.y + height, pos.z);
				gameObject.transform.LookAt(target.transform);
			}
			if (gameStatus != null) {
				gameStatus.text = target.GetComponent<petanim>().GameStatus;
			}
		}

		if (displayTimeToLeft != null) {
			displayTimeToLeft.text = "Time to Left " + target.GetComponent<petanim>().Energy.ToString("000.00");
		}
    }
}
